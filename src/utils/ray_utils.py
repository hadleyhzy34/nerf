"""Ray generation utilities for NeRF."""

import torch
import numpy as np # Though not strictly needed for get_rays itself, often used with poses

def get_rays(H, W, focal, c2w):
    """Generate rays for a given camera.

    This function computes ray origins and directions in world coordinates
    for each pixel of an image. It assumes a pinhole camera model.
    The coordinate system conventions are:
    - Image: Origin at top-left. Pixel indices (row `j`, col `i`) increase
             downwards and rightwards, respectively.
    - Camera: X-axis points right, Y-axis points down, Z-axis points from the
              camera into the scene (viewing along +Z if camera at origin looking towards +Z,
              or along -Z if using OpenGL convention where camera looks down its own -Z axis).
              This implementation uses the OpenGL convention (Y-down, X-right, camera looks along -Z),
              consistent with common NeRF datasets like Blender.
    - World: Standard right-handed system. `c2w` matrix transforms points from
             camera coordinates to world coordinates.

    Args:
        H (int): Image height in pixels.
        W (int): Image width in pixels.
        focal (float): Focal length of the camera.
        c2w (torch.Tensor): Camera-to-world transformation matrix of shape [4, 4] or [3, 4].
                            It's assumed that this tensor is on the target computation device.

    Returns:
        rays_o (torch.Tensor): Ray origins in world coordinates. Shape: [H, W, 3].
                               Same device as `c2w`.
        rays_d (torch.Tensor): Normalized ray directions in world coordinates. Shape: [H, W, 3].
                               Same device as `c2w`.
    """
    # Create a meshgrid of pixel coordinates.
    # `pixels_y` corresponds to row index (from 0 to H-1), `pixels_x` to column index (from 0 to W-1).
    # `indexing='ij'` ensures `pixels_y` is [H, W] and `pixels_x` is [H, W].
    pixels_y, pixels_x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=c2w.device),
        torch.arange(W, dtype=torch.float32, device=c2w.device),
        indexing='ij'
    )

    # Compute ray directions in camera coordinates.
    # The Z-direction is negative, following the OpenGL convention where the camera looks along its -Z axis.
    dirs_x = (pixels_x - W * 0.5) / focal
    dirs_y = (pixels_y - H * 0.5) / focal  # Y positive downwards in image, Y positive downwards in camera space
    dirs_z = -torch.ones_like(dirs_x)      # Camera looks along -Z axis in its own coordinate system

    dirs_cam = torch.stack([dirs_x, dirs_y, dirs_z], dim=-1)  # Shape: [H, W, 3]

    # Transform ray directions from camera space to world space.
    # Rotation part of c2w is c2w[:3, :3].
    # Multiply (vector) by (rotation matrix): (dirs_cam @ R.T) or (R @ dirs_cam.unsqueeze(-1)).squeeze(-1)
    # A common way is to sum product: sum(dirs_cam[..., None, :] * R_matrix, dim=-1)
    rays_d = torch.sum(dirs_cam[..., None, :] * c2w[:3, :3], dim=-1)  # Shape: [H, W, 3]

    # Normalize ray directions to be unit vectors.
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    # Ray origins are the camera's translation part in world coordinates.
    # This is the last column of the c2w matrix (c2w[:3, -1]).
    rays_o = c2w[:3, -1].expand(rays_d.shape)  # Shape: [H, W, 3]

    return rays_o.contiguous(), rays_d.contiguous()
