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
import matplotlib.pyplot as plt
from tqdm import tqdm

# Adjust imports based on actual project structure
# Assuming src is in PYTHONPATH or script is run from root with python -m scripts.train_nerf
from src.datasets.blender import BlenderDataset
from src.models.nerf import PositionalEncoder, NeRF, render_rays, sample_pdf

def parse_args():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='Path to config file.')
    parser.add_argument('--expname', type=str, required=True, help='Experiment name.')
    parser.add_argument('--basedir', type=str, default='./logs/', help='Base directory for logging and checkpoints.')

    # Dataset options
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the Blender dataset.')
    # Note: --white_bkgd is for rendering, --data_white_bkgd is for dataset loading
    parser.add_argument('--white_bkgd', action='store_true', help='For rendering, if set, composite output onto a white background.')
    parser.add_argument('--data_white_bkgd', action='store_true', help='For dataset loading, if set, blend RGBA images onto a white background. Default is black.')

    # Training options
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Number of rays per batch.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--lr_decay_factor', type=float, default=0.1, help='Factor by which to decay learning rate.')
    parser.add_argument('--lr_decay_steps', type=int, default=250000, help='Number of steps over which to decay learning rate (or epochs if using StepLR with epochs).') # Adjusted help
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader workers.')

    # NeRF model options
    parser.add_argument('--net_depth', type=int, default=8, help='Depth of the main NeRF MLP (D).')
    parser.add_argument('--net_width', type=int, default=256, help='Width of the main NeRF MLP (W).')
    parser.add_argument('--input_ch_pts_L', type=int, default=10, help='Number of frequencies for points positional encoding (L_pts).')
    parser.add_argument('--input_ch_views_L', type=int, default=4, help='Number of frequencies for view directions positional encoding (L_views).')
    parser.add_argument('--use_viewdirs', action='store_true', help='Whether to use view directions.')

    # Rendering options
    parser.add_argument('--N_samples_coarse', type=int, default=64, help='Number of coarse samples per ray.')
    parser.add_argument('--N_samples_fine', type=int, default=128, help='Number of fine samples per ray (for hierarchical sampling).')
    parser.add_argument('--use_hierarchical', action='store_true', help='Whether to use hierarchical sampling.')
    parser.add_argument('--lindisp', action='store_true', help='Sample linearly in disparity rather than depth.')
    parser.add_argument('--near', type=float, default=2.0, help='Near bound for ray sampling.')
    parser.add_argument('--far', type=float, default=6.0, help='Far bound for ray sampling.')

    # Logging/checkpointing options
    parser.add_argument('--log_freq', type=int, default=100, help='Logging frequency in iterations.')
    parser.add_argument('--save_freq', type=int, default=10, help='Checkpoint saving frequency in epochs.')
    parser.add_argument('--val_img_freq', type=int, default=500, help='Validation image rendering frequency in iterations.')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to checkpoint file to resume training from (e.g., "latest.pth" or a specific checkpoint file).')

    args = parser.parse_args()
    return args

def train():
    args = parse_args()

    # Set random seed
    if args.seed is not None:
        print(f"Setting random seed to {args.seed}")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Create experiment directory
    exp_dir = os.path.join(args.basedir, args.expname)
    os.makedirs(exp_dir, exist_ok=True)

    print(f"Experiment directory created at: {exp_dir}")

    # Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset and DataLoader
    train_dataset = BlenderDataset(dataset_path=args.dataset_path, split='train', data_white_bkgd=args.data_white_bkgd)
    # pin_memory=True can speed up CPU to GPU data transfer if CUDA is available
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    val_dataset = BlenderDataset(dataset_path=args.dataset_path, split='val', data_white_bkgd=args.data_white_bkgd)
    # For validation, batch_size is 1 (one image at a time), no shuffle, fewer workers often fine.
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Assuming H, W, focal are attributes of the dataset object after initialization
    H, W, focal = train_dataset.height, train_dataset.width, train_dataset.focal
    print(f"Dataset info: H={H}, W={W}, focal={focal}. Near plane: {args.near}, Far plane: {args.far}")
    print(f"Loaded {len(train_dataset)} training images (approx {len(train_dataset) * H * W // args.batch_size} batches), "
          f"{len(val_dataset)} validation images.")

    # Positional Encoders
    embed_fn_pts = PositionalEncoder(input_dims=3, num_freqs=args.input_ch_pts_L, log_sampling=True).to(device)
    input_ch_pts = embed_fn_pts.output_dims

    embed_fn_views = None
    input_ch_views = 0
    if args.use_viewdirs:
        embed_fn_views = PositionalEncoder(input_dims=3, num_freqs=args.input_ch_views_L, log_sampling=True).to(device)
        input_ch_views = embed_fn_views.output_dims
    print(f"Positional encoding: pts_L={args.input_ch_pts_L} (->{input_ch_pts} dims), "
          f"views_L={args.input_ch_views_L if args.use_viewdirs else 'N/A'} (->{input_ch_views} dims if used)")

    # NeRF Models
    # Skip connection index: if D=8, net_depth//2 = 4. Layer self.pts_linears[4] is the one after which concat happens.
    # This means layer self.pts_linears[5] will be Linear(W + input_ch_pts, W)
    skip_layer_index = args.net_depth // 2
    model_coarse = NeRF(D=args.net_depth, W=args.net_width,
                          input_ch_pts=input_ch_pts,
                          input_ch_views=input_ch_views,
                          skips=[skip_layer_index],
                          use_viewdirs=args.use_viewdirs).to(device)
    print(f"Coarse NeRF model initialized: D={args.net_depth}, W={args.net_width}, skips at layer {skip_layer_index}.")

    model_fine = None
    if args.use_hierarchical:
        model_fine = NeRF(D=args.net_depth, W=args.net_width,
                            input_ch_pts=input_ch_pts,
                            input_ch_views=input_ch_views,
                            skips=[skip_layer_index],
                            use_viewdirs=args.use_viewdirs).to(device)
        print(f"Fine NeRF model initialized: D={args.net_depth}, W={args.net_width}, skips at layer {skip_layer_index}.")

    # Optimizer
    params_to_optimize = list(model_coarse.parameters())
    if model_fine:
        params_to_optimize.extend(list(model_fine.parameters()))
    optimizer = optim.Adam(params_to_optimize, lr=args.lr)
    print(f"Optimizer initialized (Adam) with learning rate: {args.lr}.")

    # (Placeholder) Initialize learning rate scheduler
    # Example: Exponential decay based on global_step
    # def lr_lambda(step):
    #    return args.lr_decay_factor**(step / args.lr_decay_steps)
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    # Or, a StepLR scheduler based on epochs:
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_steps, gamma=args.lr_decay_factor) # If lr_decay_steps is in epochs
    print("Placeholder: Learning rate scheduler (e.g. LambdaLR for per-step decay or StepLR for per-epoch decay).")

    # Training State Variables
    start_epoch = 0
    global_step = 0

    # Checkpoint Loading
    if args.resume_from_checkpoint:
        checkpoint_path_arg = args.resume_from_checkpoint
        resolved_path = None

        if os.path.exists(checkpoint_path_arg):
            resolved_path = checkpoint_path_arg
        else:
            alt_path = os.path.join(args.basedir, args.expname, 'checkpoints', checkpoint_path_arg)
            if os.path.exists(alt_path):
                resolved_path = alt_path

        if resolved_path:
            print(f"Resuming training from checkpoint: {resolved_path}")
            try:
                checkpoint = torch.load(resolved_path, map_location=device)

                model_coarse.load_state_dict(checkpoint['model_coarse_state_dict'])
                if model_fine is not None and 'model_fine_state_dict' in checkpoint:
                    model_fine.load_state_dict(checkpoint['model_fine_state_dict'])
                elif model_fine is not None: # Fine model exists, but not in checkpoint
                    print("Warning: Fine model enabled, but 'model_fine_state_dict' not in checkpoint. Fine model weights are not loaded from checkpoint.")
                elif 'model_fine_state_dict' in checkpoint: # Fine model in checkpoint, but not enabled
                    print("Warning: Checkpoint contains 'model_fine_state_dict', but fine model is not currently enabled. It will not be loaded.")

                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                start_epoch = checkpoint.get('epoch', -1) + 1 # Start from NEXT epoch
                global_step = checkpoint.get('global_step', 0)

                print(f"Successfully resumed from epoch {start_epoch}, global_step {global_step}.")
                # Optionally print or compare checkpoint['args_saved_at_checkpoint']
            except Exception as e:
                print(f"Error loading checkpoint '{resolved_path}': {e}. Starting training from scratch.")
                start_epoch = 0 # Reset epoch and step if loading fails
                global_step = 0
        else:
            print(f"Checkpoint file '{checkpoint_path_arg}' not found. Starting training from scratch.")

    # SummaryWriter
    tb_writer_path = os.path.join(args.basedir, args.expname, 'tensorboard')
    os.makedirs(tb_writer_path, exist_ok=True)
    writer = SummaryWriter(tb_writer_path)
    print(f"Tensorboard logs will be saved to: {tb_writer_path}")

    # Note: start_epoch and global_step are initialized before checkpoint loading logic now.
    print("Initial training state: start_epoch={}, global_step={}".format(start_epoch, global_step))

    print("Starting training...")

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

            # Manual learning rate decay (exponential)
            current_lr = args.lr * (args.lr_decay_factor**(global_step / args.lr_decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            # Coarse pass
            coarse_results = render_rays(
                nerf_model=model_coarse, embed_fn_pts=embed_fn_pts, embed_fn_views=embed_fn_views,
                rays_o=rays_o, rays_d=rays_d, near=args.near, far=args.far,
                N_samples=args.N_samples_coarse, rand=True, lindisp=args.lindisp,
                use_viewdirs=args.use_viewdirs, white_bkgd=args.white_bkgd
            )
            loss_coarse = torch.mean((coarse_results['rgb_map'] - target_rgb)**2)
            total_loss = loss_coarse

            # Fine pass (hierarchical sampling)
            if args.use_hierarchical and model_fine:
                with torch.no_grad():
                    z_vals_coarse = coarse_results['z_vals'].detach()
                    weights_coarse = coarse_results['weights'].detach()

                    # Calculate midpoints of z_vals for PDF sampling bins
                    z_vals_mid = 0.5 * (z_vals_coarse[..., 1:] + z_vals_coarse[..., :-1])
                    # Using weights for PDF, can also use alpha values
                    # Add small epsilon to prevent issues with zero weights
                    weights_pdf = weights_coarse[..., 1:-1] + 1e-5 # Exclude first and last weight for midpoints

                    z_samples_fine = sample_pdf(bins=z_vals_mid, weights=weights_pdf,
                                                N_samples=args.N_samples_fine, det=False) # det=True for testing, False for training

                # Combine coarse and fine samples, sort them
                z_vals_combined, _ = torch.sort(torch.cat([z_vals_coarse, z_samples_fine], dim=-1), dim=-1)

                # Render with fine model using combined samples
                fine_results = render_rays(
                    nerf_model=model_fine, embed_fn_pts=embed_fn_pts, embed_fn_views=embed_fn_views,
                    rays_o=rays_o, rays_d=rays_d, near=args.near, far=args.far, # near/far might not be strictly needed if z_vals_override is full
                    N_samples=z_vals_combined.shape[-1], # This will be ignored due to z_vals_override
                    rand=True, lindisp=args.lindisp, use_viewdirs=args.use_viewdirs,
                    white_bkgd=args.white_bkgd, z_vals_override=z_vals_combined
                )
                loss_fine = torch.mean((fine_results['rgb_map'] - target_rgb)**2)
                total_loss += loss_fine

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss_sum += total_loss.item()
            global_step += 1

            # Logging
            if global_step % args.log_freq == 0:
                psnr_coarse_val = -10. * torch.log10(loss_coarse.detach()) if loss_coarse > 0 else torch.tensor(0.0)
                writer.add_scalar('Loss/total_train', total_loss.item(), global_step)
                writer.add_scalar('Loss/coarse_train', loss_coarse.item(), global_step)
                writer.add_scalar('PSNR/coarse_train', psnr_coarse_val.item(), global_step)
                writer.add_scalar('LearningRate', current_lr, global_step)

                log_message_parts = [f"GS: {global_step}", f"LR: {current_lr:.7f}",
                                     f"Loss_T: {total_loss.item():.4f}", f"PSNR_c: {psnr_coarse_val.item():.2f}"]
                if args.use_hierarchical and model_fine and 'loss_fine' in locals():
                    psnr_fine_val = -10. * torch.log10(loss_fine.detach()) if loss_fine > 0 else torch.tensor(0.0)
                    writer.add_scalar('Loss/fine_train', loss_fine.item(), global_step)
                    writer.add_scalar('PSNR/fine_train', psnr_fine_val.item(), global_step)
                    log_message_parts.extend([f"Loss_f: {loss_fine.item():.4f}", f"PSNR_f: {psnr_fine_val.item():.2f}"])

                batch_iterator.set_postfix_str(" ".join(log_message_parts), refresh=True)

            # Validation image rendering (placeholder)
            if global_step > 0 and global_step % args.val_img_freq == 0:
                tqdm.write(f"\nStep {global_step}: Rendering validation image (placeholder)...")
                # TODO: Implement validation image rendering logic
                # e.g., take a fixed image from val_loader, render it, save to imageio, log to tensorboard
                pass

        avg_epoch_loss = epoch_loss_sum / len(train_loader)
        tqdm.write(f"Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}. Global Step: {global_step}")

        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(epoch, global_step, model_coarse, model_fine, optimizer, args)

    print("Saving final checkpoint...")
    save_checkpoint(args.num_epochs - 1, global_step, model_coarse, model_fine, optimizer, args, is_final=True)

    writer.close()
    print("Training complete.")

# Define save_checkpoint function here or ensure it's imported if defined elsewhere
def save_checkpoint(epoch, global_step, model_coarse, model_fine, optimizer, args, is_final=False):
    checkpoint_dir = os.path.join(args.basedir, args.expname, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_filename = f'checkpoint_epoch_{epoch:04d}_step_{global_step:07d}.pth'
    if is_final: # Use epoch as the actual last completed epoch for final save
        checkpoint_filename = f'checkpoint_final_epoch_{epoch:04d}_step_{global_step:07d}.pth'

    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

    save_dict = {
        'epoch': epoch, # The epoch just completed
        'global_step': global_step,
        'args_saved_at_checkpoint': vars(args),
        'model_coarse_state_dict': model_coarse.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if model_fine is not None:
        save_dict['model_fine_state_dict'] = model_fine.state_dict()

    torch.save(save_dict, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    if not is_final: # Also save as 'latest.pth' for easy resume
        latest_path = os.path.join(checkpoint_dir, 'latest.pth')
        torch.save(save_dict, latest_path)
        print(f"Latest checkpoint updated at {latest_path}")


if __name__ == '__main__':
    train()
