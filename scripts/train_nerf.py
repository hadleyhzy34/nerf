"""Main training script for the NeRF model.

This script handles:
- Parsing command-line arguments and configuration files.
- Setting up datasets and dataloaders for training and validation.
- Initializing the NeRF model (coarse and optionally fine) and positional encoders.
- Setting up the optimizer and learning rate schedule.
- Managing checkpoint saving and loading for resuming training.
- The main training loop, including:
    - Ray sampling and rendering (coarse and fine passes).
    - Loss computation (MSE).
    - Backpropagation and optimizer steps.
    - Logging metrics (loss, PSNR, learning rate) to console and TensorBoard.
- Placeholder for validation image rendering.
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import configargparse
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import imageio
# import matplotlib.pyplot as plt # Not currently used, can be removed or kept for future debugging
from tqdm import tqdm

# Adjust imports based on actual project structure
from src.datasets.blender import BlenderDataset
from src.models.nerf import PositionalEncoder, NeRF, render_rays, sample_pdf
from src.utils.image_utils import mse_to_psnr # Import new utility

def parse_args():
    """Parses command-line arguments and config file settings.

    Uses configargparse to allow for defining arguments via CLI, config files,
    or environment variables.

    Returns:
        configargparse.Namespace: Parsed arguments.
    """
    parser = configargparse.ArgumentParser(
        default_config_files=['./configs/default.txt'], # Example: add a default config
        auto_env_var_prefix='NERF_', # Example: env vars like NERF_EXPNAME
        config_file_parser_class=configargparse.DefaultConfigFileParser # Allow INI-style configs
    )
    parser.add_argument('--config', is_config_file=True, help='Path to config file.')
    parser.add_argument('--expname', type=str, required=True, help='Experiment name, used for logging and saving checkpoints.')
    parser.add_argument('--basedir', type=str, default='./logs/', help='Base directory where all experiment logs and checkpoints are stored.')

    # --- Dataset Options ---
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the root directory of the Blender dataset (e.g., data/nerf_synthetic/lego).')
    parser.add_argument('--data_white_bkgd', action='store_true', help='If set, dataset images with alpha channels are blended onto a white background during loading. Default is black.')

    # --- Rendering Options (used during training for coarse/fine passes, and by white_bkgd for render_rays) ---
    parser.add_argument('--white_bkgd', action='store_true', help='If set, rendered images (during training/eval/render) are composited onto a white background if opacity < 1. Default is black.')
    parser.add_argument('--near', type=float, default=2.0, help='Near sampling bound for rays.')
    parser.add_argument('--far', type=float, default=6.0, help='Far sampling bound for rays.')
    parser.add_argument('--N_samples_coarse', type=int, default=64, help='Number of coarse samples per ray during training.')
    parser.add_argument('--N_samples_fine', type=int, default=128, help='Number of fine samples per ray for hierarchical sampling during training.')
    parser.add_argument('--use_hierarchical', action='store_true', help='If set, enables hierarchical sampling with a fine NeRF model.')
    parser.add_argument('--lindisp', action='store_true', help='If set, sample linearly in disparity along rays, otherwise sample linearly in depth.')

    # --- Training Options ---
    parser.add_argument('--num_epochs', type=int, default=200, help='Total number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Number of rays processed per batch during training.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate for the Adam optimizer.')
    parser.add_argument('--lr_decay_factor', type=float, default=0.1, help='Factor by which the learning rate is decayed.')
    parser.add_argument('--lr_decay_steps', type=int, default=250000, help='Number of global steps after which the learning rate is decayed by the factor. This is a step-based decay.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes for the DataLoader.')

    # --- NeRF Model Architecture Options ---
    parser.add_argument('--net_depth', type=int, default=8, help='Number of linear layers in the main NeRF MLP (D).')
    parser.add_argument('--net_width', type=int, default=256, help='Number of hidden units per layer in the NeRF MLP (W).')
    parser.add_argument('--input_ch_pts_L', type=int, default=10, help='Number of frequency bands (L) for positional encoding of 3D points.')
    parser.add_argument('--input_ch_views_L', type=int, default=4, help='Number of frequency bands (L) for positional encoding of view directions.')
    parser.add_argument('--use_viewdirs', action='store_true', help='If set, use view-dependent effects by providing view directions to the NeRF model.')

    # --- Logging and Checkpointing Options ---
    parser.add_argument('--log_freq', type=int, default=100, help='Frequency (in global steps) for logging training metrics to console and TensorBoard.')
    parser.add_argument('--save_freq', type=int, default=10, help='Frequency (in epochs) for saving model checkpoints.')
    parser.add_argument('--val_img_freq', type=int, default=5000, help='Frequency (in global steps) for rendering and logging a validation image (current implementation is a placeholder).')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to a checkpoint file (.pth) to resume training from. Can be "latest.pth" or a specific filename within the experiment\'s checkpoint directory.')

    args = parser.parse_args()
    return args

def save_checkpoint(epoch: int, global_step: int,
                    model_coarse: nn.Module, model_fine: nn.Module,
                    optimizer: optim.Optimizer, args: configargparse.Namespace,
                    H: int, W: int, focal: float, is_final: bool = False):
    """Saves a training checkpoint.

    Args:
        epoch (int): The epoch number just completed.
        global_step (int): The total number of iterations (global steps) completed.
        model_coarse (nn.Module): The coarse NeRF model.
        model_fine (nn.Module, optional): The fine NeRF model. Can be None.
        optimizer (optim.Optimizer): The optimizer.
        args (configargparse.Namespace): The training arguments.
        H (int): Image height used during training.
        W (int): Image width used during training.
        focal (float): Focal length used during training.
        is_final (bool, optional): If True, saves as a final checkpoint with a specific name
                                   and does not update 'latest.pth'. Defaults to False.
    """
    checkpoint_dir = os.path.join(args.basedir, args.expname, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_filename = f'checkpoint_epoch_{epoch:04d}_step_{global_step:07d}.pth'
    if is_final:
        checkpoint_filename = f'checkpoint_final_epoch_{epoch:04d}_step_{global_step:07d}.pth'

    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

    save_dict = {
        'epoch': epoch,
        'global_step': global_step,
        'args_saved_at_checkpoint': vars(args), # Save the config used for this run
        'H': H, # Save dataset/render parameters
        'W': W,
        'focal': focal,
        'model_coarse_state_dict': model_coarse.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if model_fine is not None:
        save_dict['model_fine_state_dict'] = model_fine.state_dict()

    torch.save(save_dict, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    if not is_final: # Also save as 'latest.pth' for easy resumption
        latest_path = os.path.join(checkpoint_dir, 'latest.pth')
        torch.save(save_dict, latest_path)
        # print(f"Latest checkpoint updated at {latest_path}") # Can be a bit verbose for every save


def train():
    """Main training function for NeRF.

    Orchestrates the entire training process including argument parsing,
    data loading, model initialization, training loop, logging, and checkpointing.
    """
    args = parse_args()

    # --- Setup ---
    if args.seed is not None:
        print(f"Setting random seed to {args.seed}")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    exp_dir = os.path.join(args.basedir, args.expname)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Experiment directory: {exp_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Datasets and DataLoaders ---
    train_dataset = BlenderDataset(dataset_path=args.dataset_path, split='train', data_white_bkgd=args.data_white_bkgd)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    # Validation dataset (optional, but good practice for monitoring)
    val_dataset = BlenderDataset(dataset_path=args.dataset_path, split='val', data_white_bkgd=args.data_white_bkgd)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    H, W, focal = train_dataset.height, train_dataset.width, train_dataset.focal
    if H is None or W is None or focal is None:
        raise ValueError("Dataset H, W, or focal length not loaded correctly. Check dataset path and integrity.")
    print(f"Dataset info: H={H}, W={W}, Focal={focal:.2f}. Ray bounds: Near={args.near}, Far={args.far}")
    print(f"Loaded {len(train_dataset)} training images, {len(val_dataset)} validation images.")

    # --- Models and Positional Encoders ---
    embed_fn_pts = PositionalEncoder(input_dims=3, num_freqs=args.input_ch_pts_L, log_sampling=True).to(device)
    input_ch_pts = embed_fn_pts.output_dims

    embed_fn_views = None
    input_ch_views = 0
    if args.use_viewdirs:
        embed_fn_views = PositionalEncoder(input_dims=3, num_freqs=args.input_ch_views_L, log_sampling=True).to(device)
        input_ch_views = embed_fn_views.output_dims

    print(f"Positional encoding: Points L={args.input_ch_pts_L} (->{input_ch_pts} dims), "
          f"Views L={args.input_ch_views_L if args.use_viewdirs else 'N/A'} (->{input_ch_views} dims if used)")

    skip_layer_index = args.net_depth // 2
    model_coarse = NeRF(D=args.net_depth, W=args.net_width,
                          input_ch_pts=input_ch_pts, input_ch_views=input_ch_views,
                          skips=[skip_layer_index], use_viewdirs=args.use_viewdirs).to(device)
    print(f"Coarse NeRF model: D={args.net_depth}, W={args.net_width}, Skips at layer {skip_layer_index}.")

    model_fine = None
    if args.use_hierarchical:
        model_fine = NeRF(D=args.net_depth, W=args.net_width,
                            input_ch_pts=input_ch_pts, input_ch_views=input_ch_views,
                            skips=[skip_layer_index], use_viewdirs=args.use_viewdirs).to(device)
        print(f"Fine NeRF model: D={args.net_depth}, W={args.net_width}, Skips at layer {skip_layer_index}.")

    # --- Optimizer ---
    params_to_optimize = list(model_coarse.parameters())
    if model_fine:
        params_to_optimize.extend(list(model_fine.parameters()))
    optimizer = optim.Adam(params_to_optimize, lr=args.lr)
    print(f"Optimizer: Adam, Initial LR: {args.lr}.")

    # --- Learning Rate Scheduler (Manual implementation in loop) ---
    # A more formal scheduler could be used here if desired, e.g.,
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: args.lr_decay_factor**(step / args.lr_decay_steps))
    # If using such a scheduler, remember to call scheduler.step() appropriately.
    print(f"LR schedule: Exponential decay with factor {args.lr_decay_factor} every {args.lr_decay_steps} global steps.")

    # --- Training State and Checkpoint Loading ---
    start_epoch = 0
    global_step = 0

    if args.resume_from_checkpoint:
        checkpoint_path_arg = args.resume_from_checkpoint
        # Resolve path: 1. Direct, 2. In experiment's checkpoint dir
        resolved_path = checkpoint_path_arg
        if not os.path.exists(resolved_path):
            alt_path = os.path.join(exp_dir, 'checkpoints', checkpoint_path_arg)
            if os.path.exists(alt_path):
                resolved_path = alt_path
            else: # Could not find it
                resolved_path = None

        if resolved_path and os.path.exists(resolved_path):
            print(f"Resuming training from checkpoint: {resolved_path}")
            try:
                checkpoint = torch.load(resolved_path, map_location=device)

                model_coarse.load_state_dict(checkpoint['model_coarse_state_dict'])
                if model_fine is not None and 'model_fine_state_dict' in checkpoint:
                    model_fine.load_state_dict(checkpoint['model_fine_state_dict'])
                elif model_fine is not None:
                    print("Warning: Fine model enabled, but 'model_fine_state_dict' not found in checkpoint.")
                elif 'model_fine_state_dict' in checkpoint:
                     print("Warning: Checkpoint has 'model_fine_state_dict', but fine model not enabled. Not loading it.")

                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                start_epoch = checkpoint.get('epoch', -1) + 1 # Resume from NEXT epoch
                global_step = checkpoint.get('global_step', 0)

                # Optional: Restore H, W, focal if they differ, though usually they are fixed by dataset.
                # H_ckpt = checkpoint.get('H', H) # ... etc.

                print(f"Resumed from epoch {start_epoch}, global_step {global_step}.")
                # One could also compare args saved in checkpoint with current args:
                # if 'args_saved_at_checkpoint' in checkpoint:
                #     print("Saved args:", checkpoint['args_saved_at_checkpoint'])
            except Exception as e:
                print(f"ERROR loading checkpoint '{resolved_path}': {e}. Training from scratch.")
                start_epoch = 0
                global_step = 0
        else:
            print(f"Checkpoint file '{checkpoint_path_arg}' not found. Training from scratch.")

    # --- TensorBoard Summary Writer ---
    tb_writer_path = os.path.join(exp_dir, 'tensorboard')
    os.makedirs(tb_writer_path, exist_ok=True)
    writer = SummaryWriter(tb_writer_path)
    print(f"TensorBoard logs saved to: {tb_writer_path}")

    print(f"Initial training state: start_epoch={start_epoch}, global_step={global_step}")
    print("Starting training...")

    # --- Main Training Loop ---
    for epoch in range(start_epoch, args.num_epochs):
        model_coarse.train()
        if model_fine:
            model_fine.train()

        epoch_loss_sum = 0.0
        batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}", leave=True)

        for batch_idx, data_batch in enumerate(batch_iterator):
            rays_o = data_batch['rays_o'].to(device)
            rays_d = data_batch['rays_d'].to(device)
            target_rgb = data_batch['target_rgb'].to(device)

            # Manual learning rate decay (exponential decay based on global_step)
            # This is a common way to implement step-based decay.
            # A PyTorch scheduler could also be used (e.g., LambdaLR or StepLR).
            current_lr = args.lr * (args.lr_decay_factor**(global_step / args.lr_decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            # --- Coarse Pass ---
            coarse_results = render_rays(
                nerf_model=model_coarse, embed_fn_pts=embed_fn_pts, embed_fn_views=embed_fn_views,
                rays_o=rays_o, rays_d=rays_d, near=args.near, far=args.far,
                N_samples=args.N_samples_coarse, rand=True, lindisp=args.lindisp,
                use_viewdirs=args.use_viewdirs, white_bkgd=args.white_bkgd
            )
            loss_coarse = torch.mean((coarse_results['rgb_map'] - target_rgb)**2)
            total_loss = loss_coarse

            # --- Fine Pass (Hierarchical Sampling) ---
            if args.use_hierarchical and model_fine:
                with torch.no_grad(): # Detach graph for PDF sampling inputs
                    z_vals_coarse = coarse_results['z_vals'].detach()
                    weights_coarse = coarse_results['weights'].detach()

                    # Calculate midpoints of z_vals for PDF sampling bins
                    z_vals_mid = 0.5 * (z_vals_coarse[..., 1:] + z_vals_coarse[..., :-1])
                    # Use weights from coarse pass to define PDF for sampling fine points
                    # Exclude first and last weights as they correspond to intervals beyond midpoints
                    weights_pdf = weights_coarse[..., 1:-1] + 1e-5 # Add epsilon for stability

                    # Sample new z-values based on the PDF from coarse weights
                    z_samples_fine = sample_pdf(bins=z_vals_mid, weights=weights_pdf,
                                                N_samples=args.N_samples_fine,
                                                det=False) # Stochastic sampling during training

                # Combine coarse samples and new fine samples, then sort
                z_vals_combined, _ = torch.sort(torch.cat([z_vals_coarse, z_samples_fine], dim=-1), dim=-1)

                # Render with fine model using the combined set of samples
                fine_results = render_rays(
                    nerf_model=model_fine, embed_fn_pts=embed_fn_pts, embed_fn_views=embed_fn_views,
                    rays_o=rays_o, rays_d=rays_d, near=args.near, far=args.far,
                    N_samples=z_vals_combined.shape[-1], # Will be ignored if z_vals_override is used
                    rand=True, lindisp=args.lindisp, use_viewdirs=args.use_viewdirs,
                    white_bkgd=args.white_bkgd, z_vals_override=z_vals_combined
                )
                loss_fine = torch.mean((fine_results['rgb_map'] - target_rgb)**2)
                total_loss += loss_fine # Add fine loss to total loss

            # --- Optimization Step ---
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss_sum += total_loss.item()
            global_step += 1

            # --- Logging ---
            if global_step % args.log_freq == 0:
                psnr_coarse_val = mse_to_psnr(loss_coarse.detach())
                writer.add_scalar('Loss/total_train', total_loss.item(), global_step)
                writer.add_scalar('Loss/coarse_train', loss_coarse.item(), global_step)
                writer.add_scalar('PSNR/coarse_train', psnr_coarse_val.item(), global_step)
                writer.add_scalar('LearningRate', current_lr, global_step)

                log_msg_parts = [f"GS: {global_step}", f"LR: {current_lr:.7f}",
                                 f"Loss_T: {total_loss.item():.4f}", f"PSNR_c: {psnr_coarse_val.item():.2f}"]
                if args.use_hierarchical and model_fine and 'loss_fine' in locals():
                    psnr_fine_val = mse_to_psnr(loss_fine.detach())
                    writer.add_scalar('Loss/fine_train', loss_fine.item(), global_step)
                    writer.add_scalar('PSNR/fine_train', psnr_fine_val.item(), global_step)
                    log_msg_parts.extend([f"Loss_f: {loss_fine.item():.4f}", f"PSNR_f: {psnr_fine_val.item():.2f}"])

                batch_iterator.set_postfix_str(" ".join(log_msg_parts), refresh=True)

            # --- Validation Image Rendering (Placeholder) ---
            if global_step > 0 and global_step % args.val_img_freq == 0:
                # This is a placeholder. In a full implementation, one would:
                # 1. Set models to eval mode: model_coarse.eval(), model_fine.eval()
                # 2. Get a fixed batch of data from val_loader.
                # 3. Render rays (with rand=False).
                # 4. Calculate PSNR/SSIM/LPIPS for the validation image.
                # 5. Log image and metrics to TensorBoard (writer.add_image, writer.add_scalar).
                # 6. Set models back to train mode: model_coarse.train(), model_fine.train()
                tqdm.write(f"\nStep {global_step}: Rendering validation image (placeholder)...")
                pass # Placeholder for validation image rendering

        avg_epoch_loss = epoch_loss_sum / len(train_loader)
        tqdm.write(f"Epoch {epoch+1}/{args.num_epochs} completed. Average Loss: {avg_epoch_loss:.4f}. Global Step: {global_step}")

        # --- Save Checkpoint ---
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(epoch, global_step, model_coarse, model_fine, optimizer, args, H, W, focal)

    # --- End of Training ---
    print("Saving final checkpoint...")
    save_checkpoint(args.num_epochs - 1, global_step, model_coarse, model_fine, optimizer, args, H, W, focal, is_final=True)

    writer.close()
    print("Training complete.")


if __name__ == '__main__':
    train()
