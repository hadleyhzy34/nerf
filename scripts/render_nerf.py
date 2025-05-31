"""Script for rendering novel views from a trained NeRF model.

This script loads a trained NeRF model, generates a sequence of camera poses
along a specified path (e.g., spherical spiral), renders the scene from these
novel viewpoints, and saves the output as individual frames and compiled into
a video file (e.g., MP4).
"""
import os
import torch
import numpy as np
import configargparse
import imageio
from tqdm import tqdm

# Adjust imports based on actual project structure
from nerf_project.models.nerf import PositionalEncoder, NeRF, render_rays, sample_pdf
from nerf_project.utils.ray_utils import get_rays
from nerf_project.utils.image_utils import to_uint8

def parse_render_args():
    """Parses command-line arguments for the rendering script.

    Returns:
        configargparse.Namespace: Parsed arguments for rendering configuration.
    """
    parser = configargparse.ArgumentParser(
        default_config_files=['./configs/default_render.txt'], # Example default render config
        auto_env_var_prefix='NERF_RENDER_',
        config_file_parser_class=configargparse.DefaultConfigFileParser
    )
    parser.add_argument('--config', is_config_file=True, help='Path to config file (can reuse training or eval config if relevant).')
    parser.add_argument('--expname', type=str, required=True, help='Experiment name, used for locating models if checkpoint_path is relative and for structuring output directory.')
    parser.add_argument('--basedir', type=str, default='./logs/', help='Base directory where experiments (and their checkpoints) are stored.')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Specific path to the .pth checkpoint file to load for rendering.')

    # --- Output Options ---
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save rendered frames and video. Defaults to ./render_results/{expname}/{checkpoint_name}/{render_path_type}.')

    # --- Camera Path Options ---
    parser.add_argument('--render_path_type', type=str, default='spherical_spiral', choices=['spherical_spiral', 'circle'], help='Type of camera path to generate for the novel view sequence.')
    parser.add_argument('--num_frames', type=int, default=120, help='Number of frames (camera poses) to render for the video.')
    parser.add_argument('--render_radius', type=float, default=4.0, help='Radius of the spherical or circular camera path.')
    parser.add_argument('--render_elevation_deg', type=float, default=-30.0, help='Elevation angle in degrees for circular or spherical paths (e.g., -30 looks down).')
    parser.add_argument('--render_target_height', type=float, default=0.0, help='Vertical height (Y-axis) of the target point or center of the rendering path in world units.')

    # --- Rendering Parameters ---
    # These can override values from the checkpoint if specified, or use checkpoint values as defaults.
    parser.add_argument('--render_H', type=int, default=None, help='Render height in pixels. If None, attempts to use H from checkpoint.')
    parser.add_argument('--render_W', type=int, default=None, help='Render width in pixels. If None, attempts to use W from checkpoint.')
    parser.add_argument('--render_focal', type=float, default=None, help='Render focal length. If None, attempts to use focal from checkpoint or estimates from FoV.')
    parser.add_argument('--render_near', type=float, default=None, help='Render near plane distance. If None, uses value from training or a default.')
    parser.add_argument('--render_far', type=float, default=None, help='Render far plane distance. If None, uses value from training or a default.')
    parser.add_argument('--render_white_bkgd', action='store_true', help='If set, composite rendered RGB onto a white background.')

    parser.add_argument('--eval_batch_size', type=int, default=4096, help='Number of rays to process in parallel during rendering to manage memory (similar to eval).')

    # --- Model Parameters (Fallbacks if not in checkpoint) ---
    # It's strongly recommended that these are loaded from the checkpoint to ensure consistency.
    parser.add_argument('--net_depth', type=int, default=8, help="Fallback: NeRF model depth (D).")
    parser.add_argument('--net_width', type=int, default=256, help="Fallback: NeRF model width (W).")
    parser.add_argument('--input_ch_pts_L', type=int, default=10, help="Fallback: Positional encoding L for points.")
    parser.add_argument('--input_ch_views_L', type=int, default=4, help="Fallback: Positional encoding L for views.")
    parser.add_argument('--use_viewdirs', action='store_true', help="Fallback: Use view directions in model.")
    parser.add_argument('--use_hierarchical', action='store_true', help="Fallback: Use hierarchical sampling (fine model).")
    parser.add_argument('--lindisp', action='store_true', help="Fallback: Sample linearly in disparity.")
    parser.add_argument('--N_samples_coarse', type=int, default=64, help="Fallback: Number of coarse samples per ray.")
    parser.add_argument('--N_samples_fine', type=int, default=128, help="Fallback: Number of fine samples per ray.")

    args = parser.parse_args()
    return args

def pose_spherical(theta: float, phi: float, radius: float, target_height: float = 0.0) -> torch.Tensor:
    """Generates a camera-to-world pose matrix for a spherical camera path.

    The camera is positioned on a sphere of a given radius, oriented to look
    towards a target point (assumed to be at (0, target_height, 0) in the world).

    Args:
        theta (float): Azimuthal angle in degrees (rotation around the Y-axis).
        phi (float): Polar (elevation) angle in degrees from the XZ-plane.
                     Negative values look downwards.
        radius (float): Radius of the sphere.
        target_height (float, optional): Y-coordinate of the point the camera looks at.
                                         Also effectively the height of the camera's orbit center.
                                         Defaults to 0.0.

    Returns:
        torch.Tensor: A 4x4 camera-to-world transformation matrix.
    """
    # Initial camera pose: at (0,0,radius), looking along -Z (towards origin)
    # This is a common convention for "camera distance" or "radius" parameters.
    trans_t = torch.eye(4, dtype=torch.float32)
    trans_t[2, 3] = radius

    # Rotation for elevation (phi) around the X-axis
    rot_phi_mat = torch.eye(4, dtype=torch.float32)
    phi_rad = phi / 180.0 * np.pi
    rot_phi_mat[1, 1] = np.cos(phi_rad)
    rot_phi_mat[1, 2] = -np.sin(phi_rad) # Rotate Y into -Z
    rot_phi_mat[2, 1] = np.sin(phi_rad) # Rotate Z into Y
    rot_phi_mat[2, 2] = np.cos(phi_rad)

    # Rotation for azimuth (theta) around the Y-axis
    rot_theta_mat = torch.eye(4, dtype=torch.float32)
    theta_rad = theta / 180.0 * np.pi
    rot_theta_mat[0, 0] = np.cos(theta_rad)
    rot_theta_mat[0, 2] = np.sin(theta_rad) # Rotate X into Z
    rot_theta_mat[2, 0] = -np.sin(theta_rad) # Rotate Z into -X
    rot_theta_mat[2, 2] = np.cos(theta_rad)

    # Combine transformations: R_world_cam = R_theta @ R_phi
    # The camera starts at origin, points along -Z. We first translate it out, then rotate.
    # c2w = R_theta @ R_phi @ T_radius
    # This means the camera is placed at a point on the sphere, and its -Z axis points towards origin.
    c2w = rot_theta_mat @ rot_phi_mat @ trans_t

    # Shift the entire orbit vertically by target_height.
    # The camera will orbit around (0, target_height, 0) and look at it.
    world_vertical_shift = torch.eye(4, dtype=torch.float32)
    world_vertical_shift[1, 3] = target_height

    c2w = world_vertical_shift @ c2w # Apply shift to the final pose

    return c2w


def generate_camera_path(args: configargparse.Namespace, num_frames: int) -> torch.Tensor:
    """Generates a sequence of camera poses for rendering novel views.

    Args:
        args (configargparse.Namespace): Parsed arguments containing path type and parameters.
        num_frames (int): The number of camera poses (frames) to generate.

    Returns:
        torch.Tensor: A tensor of camera-to-world matrices, shape [num_frames, 4, 4].

    Raises:
        ValueError: If an unknown `render_path_type` is specified.
    """
    poses = []
    if args.render_path_type == 'spherical_spiral' or args.render_path_type == 'circle':
        # For 'circle', elevation is fixed. 'spherical_spiral' could vary elevation too.
        # This implementation currently makes both a simple circular path at fixed elevation.
        # A true spiral would also vary phi or radius over time.
        for i in range(num_frames):
            theta = (i / num_frames) * 360.0  # Azimuth from 0 to 360 degrees
            phi = args.render_elevation_deg   # Constant elevation
            pose = pose_spherical(theta, phi, args.render_radius, args.render_target_height)
            poses.append(pose)
    else:
        raise ValueError(f"Unknown render_path_type: {args.render_path_type}")

    return torch.stack(poses)


def render_novel_views():
    """Main function to render novel views from a trained NeRF model.

    Loads model, generates camera path, renders frames, and saves them as images and a video.
    """
    args = parse_render_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Setup Output Directory ---
    if args.output_dir is None:
        checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint_path))[0]
        # Structure output by experiment, checkpoint, then path type for clarity
        args.output_dir = os.path.join('./render_results', args.expname, checkpoint_name, args.render_path_type)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Rendered frames will be saved to: {args.output_dir}")

    # --- Load Checkpoint and Configuration ---
    # Resolve checkpoint path (similar to evaluation script)
    if not os.path.isabs(args.checkpoint_path):
        ckpt_path_try1 = os.path.join(args.basedir, args.expname, 'checkpoints', args.checkpoint_path)
        # Other attempts as in evaluate.py... (simplified here for brevity, assume ckpt_path_try1 or original works)
        if os.path.exists(ckpt_path_try1): resolved_ckpt_path = ckpt_path_try1
        elif os.path.exists(args.checkpoint_path): resolved_ckpt_path = args.checkpoint_path
        else: print(f"ERROR: Checkpoint not found: {args.checkpoint_path}"); return
        args.checkpoint_path = resolved_ckpt_path

    print(f"Loading checkpoint from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    saved_args_dict = checkpoint.get('args_saved_at_checkpoint', {})
    if not isinstance(saved_args_dict, dict): saved_args_dict = vars(saved_args_dict)

    # --- Determine Model and Rendering Parameters ---
    # Prioritize checkpoint for model architecture
    net_depth = saved_args_dict.get('net_depth', args.net_depth)
    net_width = saved_args_dict.get('net_width', args.net_width)
    pts_L = saved_args_dict.get('input_ch_pts_L', args.input_ch_pts_L)
    views_L = saved_args_dict.get('input_ch_views_L', args.input_ch_views_L)
    use_viewdirs_model = saved_args_dict.get('use_viewdirs', args.use_viewdirs)
    use_hierarchical_model = saved_args_dict.get('use_hierarchical', args.use_hierarchical)

    # Rendering H, W, focal: CLI override > Checkpoint H,W,focal > Fallback/Estimate
    # Determine H
    if args.render_H is not None: H_render = args.render_H; print(f"Using CLI H: {H_render}")
    elif 'H' in checkpoint and checkpoint['H'] is not None: H_render = checkpoint['H']; print(f"Using checkpoint H: {H_render}")
    else: H_render = 400; print(f"Warning: Render H not specified or in checkpoint. Defaulting to {H_render}.")
    # Determine W
    if args.render_W is not None: W_render = args.render_W; print(f"Using CLI W: {W_render}")
    elif 'W' in checkpoint and checkpoint['W'] is not None: W_render = checkpoint['W']; print(f"Using checkpoint W: {W_render}")
    else: W_render = 400; print(f"Warning: Render W not specified or in checkpoint. Defaulting to {W_render}.")
    # Determine focal
    if args.render_focal is not None: focal_render = args.render_focal; print(f"Using CLI focal: {focal_render}")
    elif 'focal' in checkpoint and checkpoint['focal'] is not None: focal_render = checkpoint['focal']; print(f"Using checkpoint focal: {focal_render}")
    else:
        # Fallback: Estimate focal from a default FoV and current W_render
        default_fov_degrees = saved_args_dict.get('camera_angle_x_degrees', 50.0) # Check if FoV was saved
        if 'camera_angle_x' in saved_args_dict: # Radians
             default_fov_degrees = np.degrees(saved_args_dict['camera_angle_x'])
        focal_render = W_render / (2. * np.tan(0.5 * np.deg2rad(default_fov_degrees)))
        print(f"Warning: Render focal not in CLI/checkpoint. Estimating as {focal_render:.2f} (W={W_render}, FoV={default_fov_degrees:.1f} deg).")

    H_render, W_render = int(H_render), int(W_render) # Ensure integer dimensions

    # Other rendering params: CLI > Checkpoint 'args_saved_at_checkpoint' > CLI default
    render_near = args.render_near if args.render_near is not None else saved_args_dict.get('near', 2.0)
    render_far = args.render_far if args.render_far is not None else saved_args_dict.get('far', 6.0)
    render_white_bkgd = args.render_white_bkgd # CLI takes precedence for this render-specific choice

    # Params affecting rendering process, ideally from training config
    lindisp_render = saved_args_dict.get('lindisp', args.lindisp)
    N_samples_coarse_render = saved_args_dict.get('N_samples_coarse', args.N_samples_coarse)
    N_samples_fine_render = saved_args_dict.get('N_samples_fine', args.N_samples_fine)

    # --- Initialize Models and Encoders ---
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
    model_coarse.eval() # Set to evaluation mode

    model_fine = None
    if use_hierarchical_model and 'model_fine_state_dict' in checkpoint and checkpoint['model_fine_state_dict'] is not None:
        model_fine = NeRF(D=net_depth, W=net_width, input_ch_pts=input_ch_pts, input_ch_views=input_ch_views,
                          skips=[skip_layer_index], use_viewdirs=use_viewdirs_model).to(device)
        model_fine.load_state_dict(checkpoint['model_fine_state_dict'])
        model_fine.eval() # Set to evaluation mode
    elif use_hierarchical_model:
        print("Warning: Hierarchical model selected, but fine model state not in checkpoint. Rendering with coarse model only.")
        use_hierarchical_model = False # Disable if not loadable

    # --- Generate Camera Poses ---
    camera_poses = generate_camera_path(args, args.num_frames).to(device)
    print(f"Generated {len(camera_poses)} camera poses for '{args.render_path_type}' path.")

    # --- Rendering Loop ---
    frames = []
    for frame_idx, c2w_pose in enumerate(tqdm(camera_poses, desc="Rendering frames")):
        # Generate rays for the current camera pose
        rays_o, rays_d = get_rays(H_render, W_render, focal_render, c2w_pose)
        rays_o_flat = rays_o.reshape(-1, 3)
        rays_d_flat = rays_d.reshape(-1, 3)

        rendered_rgb_list = []
        num_rays_total = rays_o_flat.shape[0]

        with torch.no_grad(): # Ensure no gradients are computed
            for batch_start in range(0, num_rays_total, args.eval_batch_size):
                batch_rays_o = rays_o_flat[batch_start : batch_start + args.eval_batch_size]
                batch_rays_d = rays_d_flat[batch_start : batch_start + args.eval_batch_size]

                # Coarse pass
                coarse_results = render_rays(
                    nerf_model=model_coarse, embed_fn_pts=embed_fn_pts, embed_fn_views=embed_fn_views,
                    rays_o=batch_rays_o, rays_d=batch_rays_d, near=render_near, far=render_far,
                    N_samples=N_samples_coarse_render, rand=False, lindisp=lindisp_render, # rand=False for smooth render
                    use_viewdirs=use_viewdirs_model, white_bkgd=render_white_bkgd
                )

                final_rgb_map = coarse_results['rgb_map']

                # Fine pass if hierarchical model is used
                if use_hierarchical_model and model_fine:
                    z_vals_coarse = coarse_results['z_vals']
                    weights_coarse = coarse_results['weights']
                    z_vals_mid = 0.5 * (z_vals_coarse[..., 1:] + z_vals_coarse[..., :-1])
                    weights_pdf = weights_coarse[..., 1:-1] + 1e-5 # Stability

                    z_samples_fine = sample_pdf(bins=z_vals_mid, weights=weights_pdf,
                                                N_samples=N_samples_fine_render, det=True) # det=True for smooth render

                    z_vals_combined, _ = torch.sort(torch.cat([z_vals_coarse, z_samples_fine], dim=-1), dim=-1)

                    fine_results = render_rays(
                        nerf_model=model_fine, embed_fn_pts=embed_fn_pts, embed_fn_views=embed_fn_views,
                        rays_o=batch_rays_o, rays_d=batch_rays_d, near=render_near, far=render_far,
                        N_samples=z_vals_combined.shape[-1], # Will be ignored by z_vals_override
                        rand=False, lindisp=lindisp_render, use_viewdirs=use_viewdirs_model,
                        white_bkgd=render_white_bkgd, z_vals_override=z_vals_combined
                    )
                    final_rgb_map = fine_results['rgb_map'] # Use fine model's output

                rendered_rgb_list.append(final_rgb_map)

            rendered_rgb_full = torch.cat(rendered_rgb_list, dim=0).reshape(H_render, W_render, 3)
            rendered_rgb_np = rendered_rgb_full.cpu().numpy()
            rendered_rgb_np = np.clip(rendered_rgb_np, 0.0, 1.0) # Ensure valid [0,1] range

        current_frame_uint8 = to_uint8(rendered_rgb_np)
        frames.append(current_frame_uint8)
        # Save individual frames
        imageio.imwrite(os.path.join(args.output_dir, f'frame_{frame_idx:04d}.png'), current_frame_uint8)

    # Save all frames as a video
    video_filename = f'{args.expname}_{args.render_path_type}_frames{args.num_frames}.mp4'
    video_path = os.path.join(args.output_dir, video_filename)
    imageio.mimsave(video_path, frames, fps=30, quality=8) # Adjust fps and quality as needed
    print(f"Rendered video saved to {video_path}")

if __name__ == '__main__':
    render_novel_views()
