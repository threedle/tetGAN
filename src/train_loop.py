import torch
from torch import nn
from torch.nn.modules.loss import BCEWithLogitsLoss, MSELoss
from data import MeshFeatureDataset
from torch.utils.data import DataLoader
from nets import TetAutoEncoder, TetConvLayer, TetPoolLayer, TetInstanceNorm
from tets import TetGrid, GridSequence
from tqdm import tqdm
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


class DiscriminatorLocal(nn.Module):
    def __init__(self, grid, num_convs, dim_multiplier = 4, input_size = 1):
        super(DiscriminatorLocal, self).__init__()
        layers = []
        for i in range(num_convs):
            input_channels = input_size if i == 0 else dim_multiplier * 2 ** (i - 1)
            output_channels = 1 if i == num_convs - 1 else dim_multiplier * 2 ** i
            layers.append(TetConvLayer(grid, input_channels, output_channels))
            if i <= num_convs - 2:
                layers.append(TetInstanceNorm(output_channels))
                layers.append(nn.LeakyReLU(0.2, True))
        self.layers = nn.Sequential(*layers)

    def forward(self, batch):
        return self.layers(batch)

class DiscriminatorGlobal(nn.Module):
    def __init__(self, gs, dim_multiplier = 4, input_size=4):
        super(DiscriminatorGlobal, self).__init__()
        layers = []
        for i, grid in enumerate(gs.grids[::-1]):
            if i < len(gs.grids):
                input_channels = input_size if i == 0 else dim_multiplier * 2 ** (i - 1)
                output_channels = dim_multiplier * 2 ** i
                layers.append(TetConvLayer(grid, input_channels, output_channels))
                for _ in range(2):
                    layers.append(TetInstanceNorm(output_channels))
                    layers.append(nn.LeakyReLU(0.2, inplace=True))
                    layers.append(TetConvLayer(grid, output_channels, output_channels))
                if i < len(gs.grids) - 1:
                    layers.append(TetPoolLayer('max', gs, len(gs.grids) - i - 1))
                layers.append(TetInstanceNorm(output_channels))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(len(grid.tetrahedrons) * output_channels, 1))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, batch):
        return self.layers(batch)

def gradient_penalty(net, real, fake, constant=1.0, lambda_gp=10.0, device=torch.device('cpu')):
    alpha = torch.rand(real.shape[0], 1, device=device)
    alpha = alpha.expand(real.shape[0], real.nelement() // real.shape[0]).contiguous().view(*real.shape)
    interpolatesv = alpha * real + ((1 - alpha) * fake)
    d_interpolates = net(interpolatesv)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolatesv,
        grad_outputs=torch.ones(d_interpolates.size(), device=device), 
        create_graph=True, 
        retain_graph=True, 
        only_inputs=True
    )
    gradients = gradients[0].view(real.size(0), -1)
    gradient_penalty = ((torch.linalg.vector_norm(gradients + 1e-16, dim=1) - constant) ** 2).mean() * lambda_gp
    return gradient_penalty, gradients

def kld_loss(mean, log_variance):
    return -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp()) / mean.nelement()


def get_surface(occ, n_tens):
    device = occ.device
    temp = torch.cat((occ, torch.zeros(len(occ), 1).to(device) / 0), dim=1)
    temp = temp.index_select(1, n_tens).view(*occ.shape, -1)
    counts = temp.isnan().logical_not().sum(dim=2)
    avg = temp.nan_to_num(0).sum(dim=2) / counts
    return torch.logical_and(occ == 0, avg > 0), torch.logical_and(occ == 1, avg < 1)


def train_loop(cfg):
    gpus = [torch.device(f'cuda:{i}') for i in range(cfg['num_gpus'])]
    default_device = gpus[0] if len(gpus) > 0 else torch.device('cpu')
    init_grid = TetGrid.from_file(cfg['initial_grid_path'], disable_progress=False)
    gs = GridSequence(init_grid, cfg['subdivision_depth'], compute_dists=True)

    dataset = MeshFeatureDataset(cfg["dataset_path"])
    num_input_channels = 4
    encode_deformations = num_input_channels > 1
    pos_weight = dataset.pos_weight

    # scale deformation vectors between (-1, 1)
    deformation_scalar = dataset.deformation_scalar

    train_dataloader = DataLoader(dataset, cfg['batch_size'], shuffle=True, drop_last=False, pin_memory=True)
    d_dataloader = DataLoader(dataset, cfg['batch_size'], shuffle=True, drop_last=False, pin_memory=True)
    d_dataloader_iter = d_dataloader._get_iterator()
    
    vae = TetAutoEncoder(
        gs, cfg['dim_multiplier'], cfg['code_size'], cfg['is_variational'], encode_deformations,
        cfg['upsampling'], cfg['per_tet_latent'], cfg['per_tet_latent_size'], cfg['num_convs']
    )
    d_l = DiscriminatorLocal(
        gs.grids[-1], cfg['local_discriminator_num_convs'], cfg['local_discriminator_dim_mult'], num_input_channels,
    )
    d_g = DiscriminatorGlobal(
        gs, cfg['global_discriminator_dim_mult'], num_input_channels,
    )

    if len(gpus) > 0:
        vae = torch.nn.DataParallel(vae, device_ids=[i for i in range(cfg['num_gpus'])]).to(gpus[0])
        d_l = torch.nn.DataParallel(d_l, device_ids=[i for i in range(cfg['num_gpus'])]).to(gpus[0])
        d_g = torch.nn.DataParallel(d_g, device_ids=[i for i in range(cfg['num_gpus'])]).to(gpus[0])

    gen_opt = torch.optim.Adam(list(vae.parameters()), cfg['gen_lr'], betas=(cfg['gen_beta0'], cfg['gen_beta1']))
    dis_opt = torch.optim.Adam(list(d_l.parameters()) + list(d_g.parameters()), cfg['dis_lr'], betas=(cfg['dis_beta0'], cfg['dis_beta1']))
    bce_loss = BCEWithLogitsLoss(pos_weight=pos_weight)
    mse_loss = MSELoss()

    tb = SummaryWriter(log_dir=os.path.join(cfg['log_dir'], 'logs'))
    ea = EventAccumulator(tb.log_dir)
    ea.Reload()

    os.makedirs(os.path.join(cfg['log_dir'], 'checkpoints'))

    dist = torch.distributions.normal.Normal(
        torch.tensor(0.0, device=default_device),
        torch.tensor(1.0, device=default_device)
    )

    # When scheduling loss coefficients, we experimented relative to a particular fixed dataset size.
    # For consistency, we schedule relative to this fixed dataset size i.e. w.r.t scheduling the number
    # of epochs completed is epochs * len(dataset) / CONST
    fixed_num_per_epoch_for_scheduling = 5648
    
    print('----------------------------Begin Training----------------------------')
    for epoch in (epoch_bar := tqdm(range(cfg['num_epochs']))):
        epoch_bar.set_description('Epoch progress')
        ea.Reload()
        num_epochs_per_epoch_for_scheduling = int(fixed_num_per_epoch_for_scheduling / len(dataset))
        num_epochs_completed_for_scheduling = epoch * len(dataset) / fixed_num_per_epoch_for_scheduling

        # Adaptive gradient penalty weights to help training
        adapt_gradient_penalty_base_range = cfg['adapt_gradient_penalty_base_range'] * num_epochs_per_epoch_for_scheduling
        adapt_gradient_penalty_num_smallest = cfg['adapt_gradient_penalty_num_smallest'] * num_epochs_per_epoch_for_scheduling
        adapt_gradient_penalty_window = cfg['adapt_gradient_penalty_window'] * num_epochs_completed_for_scheduling
        if cfg['adapt_local_gradient_penalty']:
            d_l_gps = [s.value for s in ea.Scalars('dis_l_gp')] if epoch > 0 else []
            d_l_gp_base = torch.tensor(
                sorted(d_l_gps[:adapt_gradient_penalty_base_range])[:adapt_gradient_penalty_num_smallest]
            ).mean() if len(d_l_gps) >= adapt_gradient_penalty_base_range else 1
        if cfg['adapt_global_gradient_penalty']:
            d_g_gps = [s.value for s in ea.Scalars('dis_g_gp')] if epoch > 0 else []
            d_g_gp_base = torch.tensor(
                sorted(d_g_gps[:adapt_gradient_penalty_base_range])[:adapt_gradient_penalty_num_smallest]
            ).mean() if len(d_g_gps) >= adapt_gradient_penalty_base_range else 1
        
        # KLD Annealing
        # ~ Some Magic numbers found experimentally ~ #
        kld_coef = 1 - (1 - 0.01) * (0.9935 ** num_epochs_completed_for_scheduling)

        ###############################################################################################################################
        running_d_l_loss = torch.tensor(0.0, device=default_device)
        running_d_l_grad_penalty = torch.tensor(0.0, device=default_device)

        running_d_g_loss = torch.tensor(0.0, device=default_device)
        running_d_g_grad_penalty = torch.tensor(0.0, device=default_device)

        running_occ_recon_loss = torch.tensor(0.0, device=default_device)
        running_def_recon_loss = torch.tensor(0.0, device=default_device)
        running_adv_l_loss = torch.tensor(0.0, device=default_device)
        running_adv_g_loss = torch.tensor(0.0, device=default_device)
        running_kld_loss = torch.tensor(0.0, device=default_device)

        for batch in (batch_bar := tqdm(train_dataloader, leave=False)):
            batch_bar.set_description('Batch progress')
            ################### Train discriminators ###################
            d_l.train()
            for p in d_l.parameters():
                p.requires_grad_(True)
            d_g.train()
            for p in d_g.parameters():
                p.requires_grad_(True)
            for _ in (dis_bar := tqdm(range(cfg['discriminator_batches_per_generator_batch']), leave=False)):
                dis_bar.set_description('Discriminator batch progress')
                d_l.zero_grad()
                d_g.zero_grad()
                try:
                    _, occ, deformations, c_deformations = next(d_dataloader_iter)
                except Exception as e:
                    d_dataloader_iter = d_dataloader._get_iterator()
                    _, occ, deformations, c_deformations = next(d_dataloader_iter)
                
                occ, deformations, c_deformations = occ.to(default_device), deformations.to(default_device), c_deformations.to(default_device)
                deformations *= deformation_scalar
                c_deformations *= deformation_scalar
                feat = torch.cat([occ.unsqueeze(2), c_deformations], dim=-1)

                # Sample shapes from Gaussian and decode with VAE
                sample_shape = (
                    len(feat),
                    len(gs.grids[0].tetrahedrons),
                    cfg['code_size'] if cfg['per_tet_latent_size'] < 0 else cfg['per_tet_latent_size']
                ) if cfg['per_tet_latent'] else (len(feat), cfg['code_size'])
                with torch.no_grad():
                    sample = dist.sample(sample_shape=sample_shape)
                    decoded = (vae.module if cfg['num_gpus'] > 0 else vae).decode(sample)
                    vae_sample = torch.cat([
                        torch.sigmoid(decoded[:, :, :1]),
                        torch.tanh(decoded[:, :, 1:])
                    ], dim=-1)
                vae_sample.requires_grad_(True)

                ################### Local discriminator training ###################
                d_l_fake = d_l(vae_sample)
                d_l_real = d_l(feat)
                d_l_fake_loss = d_l_fake.mean()
                d_l_real_loss = -d_l_real.mean()

                # Gradient penalty computation
                if cfg['adapt_local_gradient_penalty']:
                    if len(d_l_gps) >= adapt_gradient_penalty_base_range:
                        d_l_gp_curr = sum(d_l_gps[-adapt_gradient_penalty_window:]) / adapt_gradient_penalty_window
                    else:
                        d_l_gp_curr = 1
                    d_l_gp_weight = max((d_l_gp_curr / d_l_gp_base) ** 2 , 1)
                else:
                    d_l_gp_weight = 1
                
                d_l_grad_penalty = d_l_gp_weight * gradient_penalty(d_l, feat, vae_sample, lambda_gp=cfg['gradient_penalty_coefficient'], device=default_device)[0]

                (d_l_fake_loss + d_l_real_loss + d_l_grad_penalty).backward()
                running_d_l_loss += (d_l_fake_loss.detach() + d_l_real_loss.detach()) * len(feat) / cfg['discriminator_batches_per_generator_batch']
                running_d_l_grad_penalty += d_l_grad_penalty.detach() * len(feat) / cfg['gradient_penalty_coefficient'] / cfg['discriminator_batches_per_generator_batch']

                # free up tensors
                del d_l_real, d_l_fake

                ################### Global discriminator training ###################
                d_g_fake = d_g(vae_sample)
                d_g_real = d_g(feat)
                d_g_fake_loss = d_g_fake.mean()
                d_g_real_loss = -d_g_real.mean()

                # Gradient penalty computation
                if cfg['adapt_local_gradient_penalty']:
                    if len(d_g_gps) >= adapt_gradient_penalty_base_range:
                        d_g_gp_curr = sum(d_g_gps[-adapt_gradient_penalty_window:]) / adapt_gradient_penalty_window
                    else:
                        d_g_gp_curr = 1
                    d_g_gp_weight = max((d_g_gp_curr / d_g_gp_base) ** 2 , 1)
                else:
                    d_g_gp_weight = 1
                
                d_g_grad_penalty = d_g_gp_weight * gradient_penalty(d_g, feat, vae_sample, lambda_gp=cfg['gradient_penalty_coefficient'], device=default_device)[0]

                (d_g_fake_loss + d_g_real_loss + d_g_grad_penalty).backward()
                running_d_g_loss += (d_g_fake_loss.detach() + d_g_real_loss.detach()) * len(feat) / cfg['discriminator_batches_per_generator_batch']
                running_d_g_grad_penalty += d_g_grad_penalty.detach() * len(feat) / cfg['gradient_penalty_coefficient'] / cfg['discriminator_batches_per_generator_batch']

                # free up tensors
                del d_g_real, d_g_fake, decoded, vae_sample

                dis_opt.step()
            
            ################### Train generator ###################
            vae.zero_grad()
            d_l.eval()
            d_g.eval()
            for p in d_l.parameters():
                p.requires_grad_(False)
            for p in d_g.parameters():
                p.requires_grad_(False)
            
            _, occ, deformations, c_deformations = batch
            occ, deformations, c_deformations = occ.to(default_device), deformations.to(default_device), c_deformations.to(default_device)
            deformations *= deformation_scalar
            c_deformations *= deformation_scalar
            feat = torch.cat([occ.unsqueeze(2), c_deformations], dim=-1)

            decoded, mean, log_var = vae(feat)
            vae_occ = decoded[:, :, 0]
            vae_def = (vae.module if cfg['num_gpus'] > 0 else vae).calculate_deformation(decoded)

            recon_occ_loss = bce_loss(vae_occ, occ)
            recon_def_loss = mse_loss(vae_def, deformations)
            kld = kld_loss(mean, log_var)

            (recon_occ_loss + recon_def_loss + kld_coef * kld).backward()

            running_occ_recon_loss += recon_occ_loss.detach() * len(feat)
            running_def_recon_loss += recon_def_loss.detach() * len(feat)
            running_kld_loss += kld.detach()
            
            # free tensors
            del decoded, mean, log_var, vae_occ, vae_def

            # sample batch from gaussian
            sample = dist.sample(sample_shape=sample_shape)
            decoded = (vae.module if cfg['num_gpus'] > 0 else vae).decode(sample)
            vae_sample = torch.cat([
                torch.sigmoid(decoded[:, :, :1]),
                torch.tanh(decoded[:, :, 1:]),
            ], dim=-1)

            d_l_vae_sample = d_l(vae_sample)
            d_g_vae_sample = d_g(vae_sample)
            adv_l_loss = -d_l_vae_sample.mean()
            adv_g_loss = -d_g_vae_sample.mean()
            adv_loss = (cfg['local_discriminator_coefficient'] * adv_l_loss- cfg['global_discriminator_coefficient'] * adv_g_loss)
            adv_loss.backward()

            running_adv_l_loss += adv_l_loss.detach()
            running_adv_g_loss += adv_g_loss.detach()

            # free tensors
            del decoded, vae_sample, d_l_vae_sample, d_g_vae_sample

            gen_opt.step()
        
        # logging
        tb.add_scalar('gen_occ_recon_loss', running_occ_recon_loss, epoch)
        tb.add_scalar('gen_def_recon_loss', running_def_recon_loss, epoch)
        tb.add_scalar('gen_adv_l_loss', running_adv_l_loss, epoch)
        tb.add_scalar('gen_adv_g_loss', running_adv_g_loss, epoch)
        tb.add_scalar('gen_kld_loss', running_kld_loss, epoch)
        tb.add_scalar('dis_l_loss', running_d_l_loss, epoch)
        tb.add_scalar('dis_l_gp', running_d_l_grad_penalty, epoch)
        tb.add_scalar('dis_g_loss', running_d_g_loss, epoch)
        tb.add_scalar('dis_g_gp', running_d_g_grad_penalty, epoch)
        # Why is this necessary sometimes?
        tb.flush()

        if epoch > 0 and epoch % cfg['checkpoint_frequency'] == 0:
            torch.save((vae.module if cfg['num_gpus'] > 0 else vae).state_dict(), os.path.join(cfg['log_dir'], 'checkpoints', f'gen_epoch_{epoch}.ckpt'))
            torch.save((d_l.module if cfg['num_gpus'] > 0 else d_l).state_dict(), os.path.join(cfg['log_dir'], 'checkpoints', f'dis_l_epoch_{epoch}.ckpt'))
            torch.save((d_g.module if cfg['num_gpus'] > 0 else d_g).state_dict(), os.path.join(cfg['log_dir'], 'checkpoints', f'dis_g_epoch_{epoch}.ckpt'))
