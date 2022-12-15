import argparse
import os
import torch
import tqdm
import yaml

from data import MeshFeatureDataset
from nets import TetAutoEncoder
from tets import TetGrid, GridSequence

def main():
    parser = argparse.ArgumentParser()
    # These arguments must be specified by command line
    parser.add_argument('--training_config', help='Config used for training model', type=str)
    parser.add_argument('--gpu', help='Use GPU', action='store_true')
    parser.add_argument('--checkpoint', help='Path to checkpoint file', type=str)

    parser.add_argument('--num_samples', help='Number of samples to generate', type=int, default=10)
    parser.add_argument('--out_dir', help='Directory to write samples to', type=str)
    parser.add_argument('--mesh_type', help='Generate triangle or tetrahedral meshes', type=str, choices=['triangle', 'tetrahedral'])
    parser.add_argument('--laplacian_smoothing_iterations', help='Number of iterations of laplacian smoothing', type=int)

    # Using the training config should mean these arguments are easily inherited
    # They may be overriden if desired

    # Grid
    parser.add_argument('--initial_grid_path', help='Path to .tet file for coarse grid', type=str, default=argparse.SUPPRESS)
    parser.add_argument('--subdivision_depth', help='How many times to subdivide initial grid', type=int, default=argparse.SUPPRESS)

    # Dataset
    parser.add_argument('--dataset_path', help='Path to folder containing loaded data', type=str, default=argparse.SUPPRESS)

    # Model settings
    parser.add_argument('--dim_multiplier', help='Channel multiplicative factor for conv blocks (generator)', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--code_size', help='Dimensionality of latent code; if per_tet_latent, then output size of final conv layer in encoder', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--is_variational', help='Set for variational AE', type=int, choices=[0, 1], default=argparse.SUPPRESS)
    parser.add_argument('--upsampling', help='Upsampling method', type=str, choices=['unpool', 'interpolate'], default=argparse.SUPPRESS)
    parser.add_argument('--per_tet_latent', help='Set for latent code per tetrahedron', type=int, choices=[0, 1], default=argparse.SUPPRESS)
    parser.add_argument('--per_tet_latent_size', help='Dimensionality of latent code per tet, set to -1 for same as code_size', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--num_convs', help='Number of conv layers per non-linearity (generator)', type=int, default=argparse.SUPPRESS)    

    args = parser.parse_args()
    args = vars(parser.parse_args())
    
    assert args['training_config'] is not None
    with open(args['training_config'], 'r') as f:
        cfg = yaml.safe_load(f)

    # Replace config file args with any args specified through CLI
    for key in args:
        cfg[key] = args[key]    
    
    device = torch.device('cuda') if cfg['gpu'] else torch.device('cpu')
    state_dict = torch.load(cfg['checkpoint'], map_location=device)

    init_grid = TetGrid.from_file(cfg['initial_grid_path'], disable_progress=False)
    gs = GridSequence(init_grid, cfg['subdivision_depth'], compute_dists=True)

    vae = TetAutoEncoder(gs, cfg['dim_multiplier'], cfg['code_size'], cfg['is_variational'], True, cfg['upsampling'], cfg['per_tet_latent'], cfg['per_tet_latent_size'], cfg['num_convs']).to(device)
    vae.load_state_dict(state_dict)
    
    deformation_scalar = MeshFeatureDataset(cfg["dataset_path"]).deformation_scalar
    dist = torch.distributions.normal.Normal(0, 1)
    if cfg['per_tet_latent']:
        latent_code_size = cfg['code_size'] if cfg['per_tet_latent_size'] < 0 else cfg['per_tet_latent_size']
        sample_shape = (cfg['num_samples'], len(gs.grids[0].tetrahedrons), latent_code_size)
    else:
        sample_shape = (cfg['num_samples'], cfg['code_size'])

    sample = dist.sample(sample_shape).to(device)
    decoded = vae.decode(sample)

    os.makedirs(cfg['out_dir'])
    print(f'Generating {len(decoded)} samples')
    for i in tqdm.tqdm(range(len(decoded))):
        mesh = vae.extract_mesh(decoded[i:i+1], cfg['mesh_type'], deformation_scalar, cfg['laplacian_smoothing_iterations'], disable_progress=True)
        if cfg['mesh_type'] == 'triangle':
            mesh.write_to_obj(filepath=os.path.join(cfg['out_dir'], f'{i}.obj'))
        else:
            mesh.write(os.path.join(cfg['out_dir'], f'{i}.msh'))


if __name__ == '__main__':
    main()
