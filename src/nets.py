import torch
from tets import TetGrid, GridSequence
from mesh import TriangleMesh, TetMesh
from torch import nn


# get neighbor tensor for a grid
def get_n_tens(grid: TetGrid):
    x = [
            [len(grid.tetrahedrons) for _ in range(4 - len(tet.neighbors))] +
            [tet.index] + 
            [neighbor.index for neighbor in tet.neighbors] for tet in grid.tetrahedrons 
        ]
    return torch.tensor(x).flatten()

class TetConvLayer(nn.Module):
    # for now, let's expect input_features to be a nTets x dimFeatures tensor where the features follow the same order
    # as the tet indices
    def __init__(self, grid: TetGrid, input_size: int, output_size: int):
        super(TetConvLayer, self).__init__()

        self.grid = grid
        self.num_tets = len(grid.tetrahedrons)
        self.input_size = input_size
        self.output_size = output_size
        self.conv = nn.Linear(5 * input_size, output_size)

        self.register_buffer(
            'n_tens',
            get_n_tens(self.grid)
        )

    def forward(self, batch):
        device = batch.device
        dim = 1
        # add extra zeros to be accessed by index_select
        x = torch.cat([batch, torch.zeros_like(batch.index_select(dim, torch.tensor(0, device=device)))], dim=dim)
        # reshape to nExamples x nTets x channels
        outputs = x[:, self.n_tens].reshape(len(x), self.num_tets , -1)
        return self.conv(outputs)


class TetPoolLayer(nn.Module):
    def __init__(self, pool_type: str, gs: GridSequence, grid_idx: int):
        super(TetPoolLayer, self).__init__()
        self.pool_type = pool_type
        self.gs = gs
        self.grid_idx = grid_idx

        new_grid = self.gs.grids[self.grid_idx - 1]
        self.register_buffer(
            'n_tens',
            torch.tensor([tet.tet_next_indices for tet in new_grid.tetrahedrons]).flatten().int()
        )

        if self.pool_type == 'max':
            self.pool = lambda x: torch.max(x, 2)[0]
        elif self.pool_type == 'avg':
            self.pool = lambda x: torch.mean(x, 2)
        else:
            raise ValueError('Pool type not implemented')

        self.pool_size = len(new_grid.tetrahedrons[0].tet_next_indices)

    def forward(self, batch):
        d0, d1, d2 = batch.shape
        output = batch.index_select(1, self.n_tens).reshape(d0, -1, self.pool_size, d2)
        return self.pool(output)


class TetUpsampleLayer(nn.Module):
    def __init__(self, gs: GridSequence, grid_idx: int, input_size: int, upsampling: str = "unpool"):
        super(TetUpsampleLayer, self).__init__()
        self.gs = gs
        self.grid_idx = grid_idx
        self.input_size = input_size
        self.upsampling_method = upsampling

        new_grid = self.gs.grids[self.grid_idx + 1]
        if self.upsampling_method == "unpool":
            self.register_buffer('n_tens', torch.tensor([int(tet.tet_prev_idx) for tet in new_grid.tetrahedrons]))
        elif self.upsampling_method == "interpolate":
            self.num_tets = len(new_grid.tetrahedrons)
            n_tens = get_n_tens(gs.grids[grid_idx])
            self.register_buffer(
                'n_tens',
                n_tens.view(
                    len(gs.grids[grid_idx].tetrahedrons), -1
                ).index_select(0, torch.tensor([int(tet.tet_prev_idx) for tet in new_grid.tetrahedrons])).flatten()
            )
            self.register_buffer(
                'scalars',
                torch.nn.functional.normalize(new_grid.interpolate_dists.reciprocal(), p=1, dim=1)
            )
     
    def forward(self, batch):
        device = batch.device
        dim = 1
        if self.upsampling_method == "unpool":
            output = batch.index_select(dim, self.n_tens)
        elif self.upsampling_method == "interpolate":
            output = torch.einsum('ijkl, jk -> ijl',
                torch.cat(
                    [batch, torch.zeros_like(batch.index_select(dim, torch.tensor(0, device=device)))], dim=dim
                ).index_select(dim, self.n_tens).view(len(batch), self.num_tets, -1, self.input_size),
                self.scalars
            )
        
        return output


class TetInstanceNorm(nn.Module):
    def __init__(self, input_size):
        super(TetInstanceNorm, self).__init__()
        self.input_size = input_size
        self.instance_norm = nn.InstanceNorm1d(input_size)

    def forward(self, batch):
        return self.instance_norm(batch.transpose(1, 2)).transpose(1, 2)


class TetAutoEncoder(nn.Module):
    def __init__(self, 
        gs: GridSequence, 
        dim_multiplier: int = 16, 
        code_size: int = 64, 
        is_variational: bool = True,
        encode_deformation: bool = False,
        upsampling: str = "unpool",
        per_tet_latent: bool = False,
        per_tet_latent_size: int = -1,
        num_convs: int = 4,
    ):
        super(TetAutoEncoder, self).__init__()
        if num_convs < 2:
            raise ValueError("Must use at least 2 convs per block")

        self.gs = gs
        self.input_dim = 4 if encode_deformation else 1
        self.output_dim = 4 if encode_deformation else 1
        self.dim_multiplier = dim_multiplier
        self.code_size = code_size
        self.is_variational = is_variational
        self.num_grids = len(self.gs.grids)
        self.num_tets = len(self.gs.grids[0].tetrahedrons)
        self.per_tet_latent = per_tet_latent

        if encode_deformation:
            self.register_buffer(
                'centroid_dists',
                torch.stack(
                    [
                        torch.tensor([[i, d] for i, d in dists.items()] + [[0, 0] for _ in range(gs.grids[-1].max_tets_per_v - len(dists))]) 
                        for dists in gs.grids[-1].centroid_dist_list
                    ]
                )
            )

        encoder_layers = []
        for i, grid in enumerate(self.gs.grids[::-1]):
            if i < len(self.gs.grids):
                input_channels = int(self.input_dim if i == 0 else self.dim_multiplier * 2 ** (i - 1))
                output_channels = int(self.dim_multiplier * 2 ** i if i < self.num_grids - 1 else self.code_size)
                encoder_layers.append(TetConvLayer(grid, input_channels, output_channels))
                for _ in range(num_convs - 1):
                    encoder_layers.append(TetInstanceNorm(output_channels))
                    encoder_layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
                    encoder_layers.append(TetConvLayer(grid, output_channels, output_channels))
                if i < len(self.gs.grids) - 1:
                    encoder_layers.append(TetPoolLayer('max', self.gs, self.num_grids - i - 1))
                encoder_layers.append(TetInstanceNorm(output_channels))
                encoder_layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        enc_final_layers = [nn.Linear(self.code_size, self.code_size if per_tet_latent_size < 0 else per_tet_latent_size)] if per_tet_latent else \
            [Lambda(lambda x: x.reshape(len(x), -1)), nn.Linear(self.code_size * self.num_tets, self.code_size),]

        self.encoder = nn.Sequential(
            *encoder_layers,
            *enc_final_layers
        )

        if self.is_variational:
            if not per_tet_latent or per_tet_latent_size < 0:
                encode_param_size = self.code_size
            else:
                encode_param_size = per_tet_latent_size
            self.encoder.add_module('vae-lr', nn.LeakyReLU(negative_slope=0.2, inplace=True))
            self.encode_mean = nn.Linear(in_features=encode_param_size, out_features=encode_param_size)
            self.encode_log_variance = nn.Linear(in_features=encode_param_size, out_features=encode_param_size)

        decoder_layers = []
        for i, grid in enumerate([None] + self.gs.grids, start=-1):
            if i < self.num_grids - 1:
                input_channels = int(
                                        self.code_size if i == -1 
                                        else self.dim_multiplier * 2 ** (self.num_grids - i - 2)
                                )
                output_channels = int(
                                        self.dim_multiplier * 2 ** (self.num_grids - i - 3)
                                        if i < self.num_grids - 2 
                                        else self.output_dim
                                )           
                if i >= 0:
                    decoder_layers.append(
                        TetUpsampleLayer(self.gs, i, input_channels, upsampling)
                    )
                    decoder_layers.append(TetConvLayer(self.gs.grids[i + 1], input_channels, input_channels))
                else:
                    decoder_layers.append(TetConvLayer(self.gs.grids[i + 1], input_channels, input_channels))
                for _ in range(num_convs - 2):
                    decoder_layers.append(TetInstanceNorm(input_channels))
                    decoder_layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
                    decoder_layers.append(TetConvLayer(self.gs.grids[i + 1], input_channels, input_channels))
                decoder_layers.append(TetInstanceNorm(input_channels))
                decoder_layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
                decoder_layers.append(TetConvLayer(self.gs.grids[i + 1], input_channels, output_channels))

                if i < self.num_grids - 2:
                    decoder_layers.append(TetInstanceNorm(output_channels))
                    decoder_layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        decoder_initial_layers = [
            nn.Linear(self.code_size if per_tet_latent_size < 0 else per_tet_latent_size, self.code_size)
        ] if per_tet_latent else \
            [
                nn.Linear(self.code_size, self.code_size * self.num_tets),
                Lambda(lambda x: x.reshape(len(x), self.num_tets, -1)),
            ]
        self.decoder = nn.Sequential(
            *decoder_initial_layers,
            TetInstanceNorm(self.code_size),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            *decoder_layers
        )

    def encode(self, batch, return_mean_log_and_variance = False):
        batch = self.encoder(batch)

        if not self.is_variational:
            return batch, None, None

        mean = self.encode_mean(batch)
        if self.training or return_mean_log_and_variance:
            device = batch.device
            log_variance = self.encode_log_variance(batch)
            std = torch.exp(log_variance * 0.5)
            eps = torch.normal(0, 1, mean.shape, device=device)

        if self.training:
            x = mean + std * eps
        else:
            x = mean
        
        if return_mean_log_and_variance:
            return x, mean, log_variance

        return x, None, None

    def decode(self, batch):
        return self.decoder(batch)

    def forward(self, batch):
        batch, mean, log_variance = self.encode(batch, self.is_variational)
        batch = self.decode(batch)
        return batch, mean, log_variance

    # Apply tanh to the end of the network output for deformations
    # interpolate tet deformations into vertex deformations
    def calculate_deformation(self, y):
        return (
                torch.tanh(y[:, :, 1:]).index_select(1, self.centroid_dists[:, :, 0].flatten().long()) *
                torch.nn.functional.normalize(
                    self.centroid_dists[:, :, 1].reciprocal().nan_to_num(posinf=0), p=1
                ).flatten()[None, :, None]
            ).view(len(y), *self.centroid_dists.shape[:2], -1).sum(2)

    # Code to actually extract mesh from network output
    # Expects a 1 x T x C input where T is the number of tets and C is the number of channels
    # Deformation scalar is the scalar applied to the dataset in order to scale network predicts to [-1, 1]
    def extract_mesh(self, network_output, mesh_type, deformation_scalar, smoothing_iterations):
        with torch.no_grad():
            occupancies = torch.sigmoid(network_output[:, :, 0]).cpu()
            deformations = self.calculate_deformation(network_output).squeeze().cpu() / deformation_scalar
            if mesh_type == 'triangle':
                mesh = TriangleMesh.from_winding_nums_and_grid(occupancies, self.gs.grids[-1])
                # Apply deformations
                for idx, i in mesh.surface_to_grid.items():
                    mesh.vertices[idx].coord += deformations[i]
                # Apply deformation weighted smoothing
                for _ in range(smoothing_iterations):
                    mesh.laplace_smoothing(deformations)
            elif mesh_type == 'tetrahedral':
                mesh = TetMesh.from_winding_nums_and_grid(occupancies, self.gs.grids[-1], defs=deformations, ls=smoothing_iterations)
            else:
                raise ValueError('Specify either triangular (surface) or tetrahedral (volumetric) mesh as output')
        return mesh


class Lambda(nn.Module):
    def __init__(self, function):
        super(Lambda, self).__init__()
        self.function = function

    def forward(self, x):
        return self.function(x)
