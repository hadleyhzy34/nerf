import torch
import os
import json
import numpy as np
import imageio

class BlenderDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, transform=None, data_white_bkgd=False):
        self.dataset_path = dataset_path
        self.split = split
        self.transform = transform
        self.data_white_bkgd = data_white_bkgd

        self.image_paths = []
        self.poses = []
        self.all_rgbs = [] # Store all images in memory
        self.focal = None
        self.height = None
        self.width = None

        json_path = os.path.join(self.dataset_path, f"transforms_{self.split}.json")
        with open(json_path, 'r') as f:
            meta = json.load(f)

        camera_angle_x = float(meta['camera_angle_x'])

        for frame in meta['frames']:
            img_path = os.path.join(self.dataset_path, frame['file_path'] + '.png')
            self.image_paths.append(img_path)
            self.poses.append(np.array(frame['transform_matrix']))

            img = imageio.imread(img_path)
            if self.height is None or self.width is None:
                self.height, self.width = img.shape[:2]

            image = (np.array(img) / 255.0).astype(np.float32)

            if image.shape[-1] == 4: # RGBA
                if self.data_white_bkgd:
                    # Blend RGB onto white background using alpha
                    image = image[...,:3] * image[...,-1:] + (1.0 - image[...,-1:])
                else:
                    # Blend RGB onto black background using alpha
                    image = image[...,:3] * image[...,-1:]
            # Now image is RGB
            self.all_rgbs.append(image) # Append numpy array, convert to tensor later

        self.poses = torch.from_numpy(np.array(self.poses)).float()
        # Convert list of numpy arrays to a single tensor
        self.all_rgbs = torch.from_numpy(np.array(self.all_rgbs)).float() # N_images, H, W, 3


        if self.width: # Ensure width is loaded
             self.focal = 0.5 * self.width / np.tan(0.5 * camera_angle_x)
        else:
            # Handle case where no images are loaded (e.g., empty split)
            # Or if first image failed to load, though imageio.imread would raise error
            print("Warning: Image width not available, focal length cannot be calculated.")


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        pose = self.poses[idx] # Get camera-to-world matrix
        rgb_target = self.all_rgbs[idx] # H, W, 3

        rays_o, rays_d = get_rays(self.height, self.width, self.focal, pose)

        if self.split == 'train':
            rays_o = rays_o.reshape(-1, 3) # (H*W, 3)
            rays_d = rays_d.reshape(-1, 3) # (H*W, 3)
            rgb_target = rgb_target.reshape(-1, 3) # (H*W, 3)
            return {"rays_o": rays_o, "rays_d": rays_d, "target_rgb": rgb_target}
        else: # 'val' or 'test'
            # For validation/testing, we return per-image rays and targets
            return {"rays_o": rays_o, "rays_d": rays_d, "target_rgb": rgb_target, "pose": pose}

# Helper function to generate rays
# Conventions:
# Image coordinate origin: Top-left. j (row index) increases downwards, i (column index) increases rightwards.
# Camera coordinate system: X right, Y down, Z inward (camera looks along positive Z).
# This is a common convention for NeRF. Blender's export might be different (often Z-up).
# The c2w matrices from Blender datasets are typically OpenGL convention (Y down, Z inward if camera at origin looking towards +Z).
# If c2w[:3,2] (Z-axis of camera in world coords) points towards scene, then camera looks along +Z in its own space.
# Ray directions are then (i - W/2)/f, (j - H/2)/f, 1.0
def get_rays(H, W, focal, c2w):
    """
    Generate rays for a given camera.
    Args:
        H (int): Image height.
        W (int): Image width.
        focal (float): Focal length.
        c2w (torch.Tensor): Camera-to-world transformation matrix [4, 4] or [3, 4].
    Returns:
        rays_o (torch.Tensor): Ray origins [H, W, 3].
        rays_d (torch.Tensor): Ray directions [H, W, 3].
    """
    # PyTorch meshgrid produces y-coordinates first (rows), then x-coordinates (columns)
    # Indexing='ij' makes it behave like np.meshgrid (x first, then y)
    # We want j (rows, from 0 to H-1) and i (cols, from 0 to W-1)
    j, i = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing='xy')
    # Transpose i, j to match image layout (H, W) after meshgrid gives (W,H) if indexing='xy'
    # Actually, indexing='ij' is what we want for (H,W) for j and (H,W) for i. Let's use default indexing='xy' then transpose.
    # Or simpler: use indexing='ij', then j is (H,W) and i is (H,W)

    # Using indexing='ij' for torch.meshgrid:
    # ii, jj = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H), indexing='ij')
    # This results in ii of shape (W,H) and jj of shape (W,H)
    # Let's use indexing='xy' and be careful:
    # i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H), indexing='xy')
    # i will be (W, H), j will be (W, H). Transpose them to (H, W)
    # i = i.transpose(0,1)
    # j = j.transpose(0,1)

    # Let's try again:
    # We need i to go from 0 to W-1 for each row, and j to go from 0 to H-1 for each column.
    # torch.meshgrid(torch.arange(W), torch.arange(H), indexing='ij') -> i (W,H), j (W,H)
    # torch.meshgrid(torch.arange(H), torch.arange(W), indexing='xy') -> j (H,W), i (H,W) This seems right.

    # Let's be explicit:
    pixels_y, pixels_x = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing='ij')
    # pixels_x: (H, W), values from 0 to W-1
    # pixels_y: (H, W), values from 0 to H-1

    # Camera coordinate directions
    # (x,y,z) in camera space: x right, y down, z forward (into scene)
    # Pixel (i,j) corresponds to camera coordinates ( (i - W/2), (j - H/2) ) on the image plane.
    # The ray direction is then ((i - W/2)/f, (j - H/2)/f, 1) for a camera looking along +Z.
    # If camera looks along -Z (OpenGL convention), then ((i - W/2)/f, (j - H/2)/f, -1).
    # The Blender dataset c2w matrices typically transform from OpenGL camera coords (Y-down, X-right, Camera looking -Z) to world.

    dirs_x = (pixels_x - W * 0.5) / focal
    dirs_y = (pixels_y - H * 0.5) / focal # Y positive downwards in image, Y positive downwards in camera
    dirs_z = -torch.ones_like(dirs_x) # Camera looks along -Z axis in its own coordinate system

    dirs_cam = torch.stack([dirs_x, dirs_y, dirs_z], dim=-1) # Shape: [H, W, 3]

    # Transform directions from camera to world space
    # c2w is [4, 4]. Rotation part is c2w[:3, :3]
    # rays_d = dirs_cam @ c2w[:3, :3].T # This would be if dirs_cam was a row vector
    # Correct way: R * d_cam where d_cam is a column vector.
    # Here dirs_cam is (H,W,3). We want to transform each (3,) vector.
    # So, (R @ d.T).T = d @ R.T
    # rays_d = torch.einsum('hwk,jk->hwj', dirs_cam, c2w[:3,:3]) # k is camera coord index, j is world coord index
    # Or simply:
    rays_d = torch.sum(dirs_cam[..., None, :] * c2w[:3, :3], dim=-1) # dirs_cam (H,W,1,3) * R (3,3) -> sum over last dim

    # Normalize ray directions
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    # Ray origins are the camera center in world coordinates
    rays_o = c2w[:3, -1].expand(rays_d.shape) # c2w[:3, -1] is (3,), expand to (H, W, 3)

    return rays_o.contiguous(), rays_d.contiguous()
