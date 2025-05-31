"""Script for evaluating a trained NeRF model.

This script loads a trained NeRF model from a checkpoint, renders images from a
specified dataset split (e.g., test or validation), and computes quantitative
evaluation metrics: Peak Signal-to-Noise Ratio (PSNR), Structural Similarity
Index (SSIM), and Learned Perceptual Image Patch Similarity (LPIPS).
Rendered images and a JSON summary of metrics are saved to an output directory.
"""
import os
import torch
import numpy as np
import configargparse
import imageio
# import matplotlib.pyplot as plt # Not currently used
from tqdm import tqdm
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity
import lpips # Requires pip install lpips
import json

# Adjust imports based on actual project structure
from src.datasets.blender import BlenderDataset
from src.models.nerf import PositionalEncoder, NeRF, render_rays, sample_pdf
from src.utils.image_utils import mse_to_psnr, to_uint8


def parse_eval_args():
    """Parses command-line arguments for the evaluation script.

    Returns:
        configargparse.Namespace: Parsed arguments, including settings for checkpoint
                                  paths, dataset, output, and rendering parameters.
    """
    parser = configargparse.ArgumentParser(
        default_config_files=['./configs/default_eval.txt'], # Example default eval config
        auto_env_var_prefix='NERF_EVAL_',
        config_file_parser_class=configargparse.DefaultConfigFileParser
    )
    parser.add_argument('--config', is_config_file=True, help='Path to config file (can reuse training config or define a specific eval config).')
    parser.add_argument('--expname', type=str, required=True, help='Experiment name, used for locating models if checkpoint_path is relative and for structuring output directory.')
    parser.add_argument('--basedir', type=str, default='./logs/', help='Base directory where experiments (and their checkpoints) are stored.')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Specific path to the .pth checkpoint file to evaluate.')

    # --- Dataset Options ---
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the root directory of the Blender dataset.')
    parser.add_argument('--dataset_split', type=str, default='test', choices=['train', 'val', 'test'], help='Which dataset split to evaluate on.')

    # --- Output Options ---
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save rendered images and metrics. Defaults to ./eval_results/{expname}/{checkpoint_name}.')

    # --- Model and Rendering Arguments ---
    # These arguments often serve as fallbacks if not found in the checkpoint's saved arguments.
    # For evaluation, it's crucial to use the same model architecture as during training.
    # Rendering parameters like N_samples can be overridden for evaluation if desired.
    parser.add_argument('--net_depth', type=int, default=8, help='Depth of the main NeRF MLP (D). Used if not in checkpoint.')
    parser.add_argument('--net_width', type=int, default=256, help='Width of the main NeRF MLP (W). Used if not in checkpoint.')
    parser.add_argument('--input_ch_pts_L', type=int, default=10, help='L for points positional encoding. Used if not in checkpoint.')
    parser.add_argument('--input_ch_views_L', type=int, default=4, help='L for views positional encoding. Used if not in checkpoint.')
    parser.add_argument('--use_viewdirs', action='store_true', help='Use view-dependent effects. Should match training setup, loaded from checkpoint if possible.')

    parser.add_argument('--N_samples_coarse', type=int, default=64, help='Number of coarse samples per ray for rendering.')
    parser.add_argument('--N_samples_fine', type=int, default=128, help='Number of fine samples per ray for hierarchical rendering.')
    parser.add_argument('--use_hierarchical', action='store_true', help='Use hierarchical sampling. Should match training setup, loaded from checkpoint if possible.')

    parser.add_argument('--lindisp', action='store_true', help='Sample linearly in disparity. Should match training setup, loaded from checkpoint if possible.')
    parser.add_argument('--white_bkgd', action='store_true', help='Composite rendered RGB onto a white background if ray opacity < 1.')
    parser.add_argument('--data_white_bkgd', action='store_true', help='Indicates if dataset images were originally blended on a white background. Loaded from checkpoint if possible.')

    parser.add_argument('--eval_batch_size', type=int, default=4096, help='Number of rays to process in parallel during rendering to manage memory.')

    args = parser.parse_args()
    return args

def normalize_for_lpips(img_tensor_chw: torch.Tensor) -> torch.Tensor:
    """Normalizes an image tensor from [0,1] range to [-1,1] for LPIPS.

    Args:
        img_tensor_chw (torch.Tensor): Image tensor with shape [C, H, W] or [N, C, H, W]
                                       and values in the range [0, 1].

    Returns:
        torch.Tensor: Normalized image tensor with values in the range [-1, 1].
    """
    return img_tensor_chw * 2. - 1.

def evaluate():
    """Main evaluation function.

    Loads a trained NeRF model, renders images from the specified dataset split,
    computes PSNR, SSIM, and LPIPS metrics, and saves results.
    """
    args = parse_eval_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Setup Output Directory ---
    if args.output_dir is None:
        checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint_path))[0]
        args.output_dir = os.path.join('./eval_results', args.expname, checkpoint_name)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Evaluation results will be saved to: {args.output_dir}")

    # --- Load Checkpoint and Model Configuration ---
    # Resolve checkpoint path if it's relative
    if not os.path.isabs(args.checkpoint_path):
        ckpt_path_try1 = os.path.join(args.basedir, args.expname, 'checkpoints', args.checkpoint_path)
        ckpt_path_try2 = os.path.join(args.basedir, args.checkpoint_path) # If expname might be part of checkpoint_path

        if os.path.exists(ckpt_path_try1):
            resolved_ckpt_path = ckpt_path_try1
        elif os.path.exists(ckpt_path_try2):
            resolved_ckpt_path = ckpt_path_try2
        elif os.path.exists(args.checkpoint_path): # Check original path as is
            resolved_ckpt_path = args.checkpoint_path
        else:
            print(f"ERROR: Checkpoint not found at '{args.checkpoint_path}' or derived paths.")
            return
        args.checkpoint_path = resolved_ckpt_path

    print(f"Loading checkpoint from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    # Load arguments saved at checkpoint to ensure model consistency
    saved_args_dict = checkpoint.get('args_saved_at_checkpoint', {})
    if not isinstance(saved_args_dict, dict): # Handles if Namespace was saved directly
        saved_args_dict = vars(saved_args_dict)
    if not saved_args_dict:
        print("Warning: Checkpoint does not contain 'args_saved_at_checkpoint'. "
              "Relying on CLI arguments for model architecture, which may be incorrect.")

    # Prioritize parameters from the checkpoint for model architecture
    # CLI arguments can override rendering details (N_samples, batch_size) for evaluation flexibility
    net_depth = saved_args_dict.get('net_depth', args.net_depth)
    net_width = saved_args_dict.get('net_width', args.net_width)
    pts_L = saved_args_dict.get('input_ch_pts_L', args.input_ch_pts_L)
    views_L = saved_args_dict.get('input_ch_views_L', args.input_ch_views_L)
    use_viewdirs_model = saved_args_dict.get('use_viewdirs', args.use_viewdirs)
    use_hierarchical_model = saved_args_dict.get('use_hierarchical', args.use_hierarchical)
    # Dataset related args from training that affect rendering or data interpretation
    data_white_bkgd_model = saved_args_dict.get('data_white_bkgd', args.data_white_bkgd)
    train_near = saved_args_dict.get('near', args.near) # Near/far used during training
    train_far = saved_args_dict.get('far', args.far)
    train_lindisp = saved_args_dict.get('lindisp', args.lindisp)

    print("--- Model Configuration (Loaded from Checkpoint / CLI Fallbacks) ---")
    print(f"  Net Depth: {net_depth}, Net Width: {net_width}")
    print(f"  Positional Encoding: Points L={pts_L}, Views L={views_L} (Viewdirs: {use_viewdirs_model})")
    print(f"  Hierarchical Sampling: {use_hierarchical_model}")
    print(f"  Training Data Background: {'White' if data_white_bkgd_model else 'Black'}")
    print(f"  Training Ray Bounds: Near={train_near}, Far={train_far}, Lindisp={train_lindisp}")
    print("--------------------------------------------------------------------")

    # --- Initialize Models and Positional Encoders ---
    embed_fn_pts = PositionalEncoder(input_dims=3, num_freqs=pts_L, log_sampling=True).to(device)
    input_ch_pts = embed_fn_pts.output_dims

    embed_fn_views = None
    input_ch_views = 0
    if use_viewdirs_model:
        embed_fn_views = PositionalEncoder(input_dims=3, num_freqs=views_L, log_sampling=True).to(device)
        input_ch_views = embed_fn_views.output_dims

    skip_layer_index = net_depth // 2
    model_coarse = NeRF(D=net_depth, W=net_width, input_ch_pts=input_ch_pts, input_ch_views=input_ch_views,
                          skips=[skip_layer_index], use_viewdirs=use_viewdirs_model).to(device)
    model_coarse.load_state_dict(checkpoint['model_coarse_state_dict'])
    model_coarse.eval() # Set model to evaluation mode
    print("Coarse model loaded and set to eval mode.")

    model_fine = None
    if use_hierarchical_model:
        if 'model_fine_state_dict' in checkpoint and checkpoint['model_fine_state_dict'] is not None:
            model_fine = NeRF(D=net_depth, W=net_width, input_ch_pts=input_ch_pts, input_ch_views=input_ch_views,
                              skips=[skip_layer_index], use_viewdirs=use_viewdirs_model).to(device)
            model_fine.load_state_dict(checkpoint['model_fine_state_dict'])
            model_fine.eval() # Set model to evaluation mode
            print("Fine model loaded and set to eval mode.")
        else:
            print("Warning: Hierarchical sampling was enabled, but fine model state not found in checkpoint. Evaluating with coarse model only.")
            use_hierarchical_model = False # Cannot use fine model if not loaded

    # --- Initialize LPIPS Metric ---
    lpips_fn = lpips.LPIPS(net='alex', version='0.1').to(device) # Using AlexNet for LPIPS

    # --- Initialize Dataset and DataLoader for Evaluation ---
    eval_dataset = BlenderDataset(dataset_path=args.dataset_path, split=args.dataset_split,
                                  data_white_bkgd=data_white_bkgd_model) # Use training data_white_bkgd for consistency
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False) # batch_size=1 for image-by-image evaluation

    # Use H, W, focal from checkpoint if available (saved from training dataset), else from eval_dataset
    H = checkpoint.get('H', eval_dataset.height)
    W = checkpoint.get('W', eval_dataset.width)
    focal = checkpoint.get('focal', eval_dataset.focal)
    if H is None or W is None or focal is None:
        raise ValueError("Image H, W, or focal not available from checkpoint or dataset. Ensure dataset is loaded correctly or H,W,focal are in checkpoint.")

    print(f"Evaluating on '{args.dataset_split}' split ({len(eval_dataset)} images). Image size: {H}x{W}, Focal: {focal:.2f}")

    # --- Evaluation Loop ---
    all_psnrs, all_ssims, all_lpips_scores = [], [], []

    for i, data_batch in enumerate(tqdm(eval_loader, desc="Evaluating images")):
        # Data from loader is for a full image (batch_size=1)
        rays_o_full = data_batch['rays_o'].squeeze(0).to(device) # Remove batch_dim, shape [H, W, 3]
        rays_d_full = data_batch['rays_d'].squeeze(0).to(device) # Shape [H, W, 3]
        target_rgb_full = data_batch['target_rgb'].squeeze(0).to(device) # Shape [H, W, 3]

        rendered_rgb_list = []
        num_pixels = H * W

        # Flatten rays for batch processing
        rays_o_flat = rays_o_full.reshape(-1, 3)
        rays_d_flat = rays_d_full.reshape(-1, 3)

        with torch.no_grad(): # Ensure no gradients are computed during evaluation
            for batch_start in range(0, num_pixels, args.eval_batch_size):
                batch_rays_o = rays_o_flat[batch_start : batch_start + args.eval_batch_size]
                batch_rays_d = rays_d_flat[batch_start : batch_start + args.eval_batch_size]

                # Coarse pass rendering
                # Use CLI args for N_samples, near/far for rendering, but model's lindisp/white_bkgd
                coarse_results = render_rays(
                    nerf_model=model_coarse, embed_fn_pts=embed_fn_pts, embed_fn_views=embed_fn_views,
                    rays_o=batch_rays_o, rays_d=batch_rays_d,
                    near=args.near, far=args.far, # Use CLI near/far for evaluation flexibility
                    N_samples=args.N_samples_coarse, rand=False, lindisp=train_lindisp, # rand=False for deterministic eval
                    use_viewdirs=use_viewdirs_model, white_bkgd=args.white_bkgd # CLI white_bkgd for render compositing
                )

                current_rgb_map = coarse_results['rgb_map']

                # Fine pass rendering if hierarchical model is used and loaded
                if use_hierarchical_model and model_fine:
                    z_vals_coarse = coarse_results['z_vals']
                    weights_coarse = coarse_results['weights']

                    # Sample points for fine pass based on coarse weights
                    z_vals_mid = 0.5 * (z_vals_coarse[..., 1:] + z_vals_coarse[..., :-1])
                    weights_pdf = weights_coarse[..., 1:-1] + 1e-5 # Add epsilon for stability

                    z_samples_fine = sample_pdf(bins=z_vals_mid, weights=weights_pdf,
                                                N_samples=args.N_samples_fine, det=True) # det=True for deterministic eval

                    # Combine coarse and fine samples, sort them
                    z_vals_combined, _ = torch.sort(torch.cat([z_vals_coarse, z_samples_fine], dim=-1), dim=-1)

                    # Render with fine model
                    fine_results = render_rays(
                        nerf_model=model_fine, embed_fn_pts=embed_fn_pts, embed_fn_views=embed_fn_views,
                        rays_o=batch_rays_o, rays_d=batch_rays_d,
                        near=args.near, far=args.far,
                        N_samples=z_vals_combined.shape[-1], # N_samples will be ignored due to z_vals_override
                        rand=False, lindisp=train_lindisp, use_viewdirs=use_viewdirs_model,
                        white_bkgd=args.white_bkgd, z_vals_override=z_vals_combined
                    )
                    current_rgb_map = fine_results['rgb_map'] # Use fine model's output

                rendered_rgb_list.append(current_rgb_map)

            # Concatenate results from all ray batches and reshape to image dimensions
            rendered_rgb_full = torch.cat(rendered_rgb_list, dim=0).reshape(H, W, 3)

        # --- Calculate Metrics ---
        rendered_rgb_np = rendered_rgb_full.cpu().numpy()
        target_rgb_np = target_rgb_full.cpu().numpy()

        # Clip to [0,1] just in case, though sigmoid should handle this for RGB
        rendered_rgb_np = np.clip(rendered_rgb_np, 0.0, 1.0)
        target_rgb_np = np.clip(target_rgb_np, 0.0, 1.0)

        # PSNR
        mse_val = np.mean((rendered_rgb_np - target_rgb_np)**2)
        psnr = mse_to_psnr(mse_val)

        # SSIM
        ssim = structural_similarity(rendered_rgb_np, target_rgb_np, multichannel=True, channel_axis=-1, data_range=1.0)

        # LPIPS (requires CHW format, normalized to [-1,1])
        img_rendered_lpips = normalize_for_lpips(torch.from_numpy(rendered_rgb_np).permute(2,0,1).unsqueeze(0)).to(device)
        img_target_lpips = normalize_for_lpips(torch.from_numpy(target_rgb_np).permute(2,0,1).unsqueeze(0)).to(device)
        lpips_score = lpips_fn(img_rendered_lpips, img_target_lpips).item()

        all_psnrs.append(psnr)
        all_ssims.append(ssim)
        all_lpips_scores.append(lpips_score)

        tqdm.write(f"Img {i+1}/{len(eval_dataset)}: PSNR={psnr:.2f}, SSIM={ssim:.4f}, LPIPS={lpips_score:.4f}")

        # --- Save Rendered Images and Ground Truth ---
        imageio.imwrite(os.path.join(args.output_dir, f'rendered_{i:03d}.png'), to_uint8(rendered_rgb_np))
        # Save GT only once if it doesn't exist (useful if running multiple evals on same dataset)
        gt_path = os.path.join(args.output_dir, f'gt_{i:03d}.png')
        if not os.path.exists(gt_path):
             imageio.imwrite(gt_path, to_uint8(target_rgb_np))

    # --- Report and Save Aggregate Results ---
    mean_psnr = np.mean(all_psnrs)
    mean_ssim = np.mean(all_ssims)
    mean_lpips = np.mean(all_lpips_scores)

    print(f"\n--- Evaluation Summary ({args.dataset_split} split) ---")
    print(f"  Mean PSNR:  {mean_psnr:.2f}")
    print(f"  Mean SSIM:  {mean_ssim:.4f}")
    print(f"  Mean LPIPS: {mean_lpips:.4f}")
    print(f"----------------------------------")

    metrics_dict = {
        'checkpoint_path': args.checkpoint_path,
        'dataset_split': args.dataset_split,
        'mean_psnr': mean_psnr,
        'mean_ssim': mean_ssim,
        'mean_lpips': mean_lpips,
        'individual_psnr': all_psnrs,
        'individual_ssim': all_ssims,
        'individual_lpips': all_lpips_scores
    }
    metrics_file_path = os.path.join(args.output_dir, 'metrics_summary.json')
    with open(metrics_file_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)

    print(f"Metrics summary saved to: {metrics_file_path}")


if __name__ == '__main__':
    evaluate()
