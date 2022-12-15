import argparse
import os
import yaml

from train_loop import train_loop

def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to config file', type=str, default='../configs/example_config.yml')
    parser.add_argument('--num_gpus', help='Number of GPUs to use', type=int, default=argparse.SUPPRESS)

    # Input paths
    parser.add_argument('--initial_grid_path', help='Path to .tet file for coarse grid', type=str, default=argparse.SUPPRESS)
    parser.add_argument('--dataset_path', help='Path to folder containing loaded data', type=str, default=argparse.SUPPRESS)

    # Logging
    parser.add_argument('--log_dir', help='Directory to log tensorboard output to', type=str, default=argparse.SUPPRESS)

    # Data settings
    parser.add_argument('--subdivision_depth', help='How many times to subdivide initial grid', type=int, default=argparse.SUPPRESS)

    # (V)AE settings
    parser.add_argument('--dim_multiplier', help='Channel multiplicative factor for conv blocks (generator)', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--code_size', help='Dimensionality of latent code; if per_tet_latent, then output size of final conv layer in encoder', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--is_variational', help='Set for variational AE', type=int, choices=[0, 1], default=argparse.SUPPRESS)
    parser.add_argument('--upsampling', help='Upsampling method', type=str, choices=['unpool', 'interpolate'], default=argparse.SUPPRESS)
    parser.add_argument('--per_tet_latent', help='Set for latent code per tetrahedron', type=int, choices=[0, 1], default=argparse.SUPPRESS)
    parser.add_argument('--per_tet_latent_size', help='Dimensionality of latent code per tet, set to -1 for same as code_size', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--num_convs', help='Number of conv layers per non-linearity (generator)', type=int, default=argparse.SUPPRESS)

    # Discriminator settings
    parser.add_argument('--local_discriminator_num_convs', help='Number of conv layers (local discriminator)', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--local_discriminator_dim_mult', help='Channel multiplicative factor for conv blocks (local discriminator)', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--global_discriminator_dim_mult', help='Channel multiplicative factor for conv blocks (global discriminator)', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--local_discriminator_coefficient', help='Weight for local adversarial loss', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--global_discriminator_coefficient', help='Weight for global adversarial loss', type=float, default=argparse.SUPPRESS)
    
    # Gradient Penalty settings
    parser.add_argument('--gradient_penalty_coefficient', help='Base weight for GP loss', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--adapt_local_gradient_penalty', help='Set to adapt local GP weight', type=int, choices=[0, 1], default=argparse.SUPPRESS)
    parser.add_argument('--adapt_global_gradient_penalty', help='Set to adapt global GP weight', type=int, choices=[0, 1], default=argparse.SUPPRESS)
    parser.add_argument('--adapt_gradient_penalty_base_range', help='Epoch to compute base GP value from', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--adapt_gradient_penalty_num_smallest', help='Number of values to compute base GP value from', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--adapt_gradient_penalty_window', help='Length of sliding window to compare to base GP to adapt GP weight', type=int, default=argparse.SUPPRESS)
    
    # Training settings
    parser.add_argument('--num_epochs', help='Number of epochs to train for', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--batch_size',  help='Number of meshes per batch', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--discriminator_batches_per_generator_batch', help='How many batches to pass to discriminators per generator batch', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--gen_lr', help='Generator learning rate', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--gen_beta0', help='Generator Adam beta0 param', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--gen_beta1', help='Generator Adam beta1 param', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--dis_lr', help='Discriminators learning rate', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--dis_beta0', help='Discriminators Adam beta0 param', type=float, default=argparse.SUPPRESS)
    parser.add_argument('--dis_beta1', help='Discriminators Adam beta1 param', type=float, default=argparse.SUPPRESS)

    args = vars(parser.parse_args())
    assert args['config'] is not None

    with open(args['config'], 'r') as f:
        cfg = yaml.safe_load(f)

    # Replace config file args with any args specified through CLI
    for key in args:
        cfg[key] = args[key]

    train_loop(cfg)
    print("Done")

if __name__ == '__main__':
    main()