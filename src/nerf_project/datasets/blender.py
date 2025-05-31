"""Module for Blender dataset loading and preprocessing.

Provides a PyTorch Dataset class to interface with the NeRF Blender dataset format,
which consists of images and JSON files describing camera transforms and intrinsics.
"""

import torch
import os
import json
import numpy as np
import imageio
from nerf_project.utils.ray_utils import get_rays # Import the moved function

class BlenderDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for loading data from the NeRF Blender dataset format.

    Handles parsing of transform JSON files (e.g., 'transforms_train.json'),
    loading images, applying alpha blending if necessary, and caching data in memory.
    It also calculates the camera focal length from the camera field of view.

    Attributes:
        dataset_path (str): Path to the root directory of the dataset (e.g., './data/lego').
        split (str): Dataset split, one of ['train', 'val', 'test'].
        transform (callable, optional): Optional transform to be applied to the data. Default is None.
        data_white_bkgd (bool): If True, RGBA images are blended onto a white background.
                                If False, they are blended onto a black background.
        image_paths (list[str]): List of paths to image files.
        poses (torch.Tensor): Tensor of camera-to-world transformation matrices [N, 4, 4].
        all_rgbs (torch.Tensor): Tensor of all loaded RGB images [N, H, W, 3].
        height (int): Height of the images in pixels.
        width (int): Width of the images in pixels.
        focal (float): Calculated focal length of the camera.
    """
    def __init__(self, dataset_path, split, transform=None, data_white_bkgd=False):
        """Initializes the BlenderDataset.

        Args:
            dataset_path (str): Path to the dataset directory (e.g., 'data/nerf_synthetic/lego').
            split (str): The dataset split to load ('train', 'val', or 'test').
            transform (callable, optional): A function/transform to apply to the data. Defaults to None.
            data_white_bkgd (bool, optional): Specifies if RGBA images should be blended
                onto a white background. If False, blends onto black. Defaults to False.
        """
        self.dataset_path = dataset_path
        self.split = split
        self.transform = transform
        self.data_white_bkgd = data_white_bkgd

        self.image_paths = []
        self.poses = []
        self.all_rgbs = [] # Store all images in memory initially as numpy arrays
        self.focal = None
        self.height = None
        self.width = None

        json_path = os.path.join(self.dataset_path, f"transforms_{self.split}.json")

        try:
            with open(json_path, 'r') as f:
                meta = json.load(f)
        except FileNotFoundError:
            print(f"Error: JSON file not found at {json_path}")
            raise

        camera_angle_x = float(meta['camera_angle_x'])

        for frame_idx, frame in enumerate(meta['frames']):
            img_path_relative = frame['file_path']
            # Ensure consistent path joining, especially if file_path might start with './'
            if img_path_relative.startswith('./'):
                img_path_relative = img_path_relative[2:]
            img_path = os.path.join(self.dataset_path, img_path_relative + '.png')

            self.image_paths.append(img_path)
            self.poses.append(np.array(frame['transform_matrix']))

            try:
                img_data = imageio.imread(img_path)
            except FileNotFoundError:
                print(f"Error: Image file not found at {img_path}")
                # Optionally, skip this frame or raise an error
                # For now, let's skip if an image is missing, and print a warning.
                print(f"Warning: Skipping frame {frame_idx} due to missing image: {img_path}")
                self.image_paths.pop()
                self.poses.pop()
                continue

            if self.height is None or self.width is None:
                self.height, self.width = img_data.shape[:2]

            # Normalize image data to [0, 1] range and convert to float32
            image = (np.array(img_data) / 255.0).astype(np.float32)

            # Handle RGBA images: blend onto a specified background color
            if image.shape[-1] == 4: # Check if alpha channel exists
                alpha_channel = image[..., -1:] # Shape: [H, W, 1]
                rgb_channels = image[..., :3]   # Shape: [H, W, 3]
                if self.data_white_bkgd:
                    # Blend RGB onto white background using alpha
                    image = rgb_channels * alpha_channel + (1.0 - alpha_channel) # White is (1,1,1)
                else:
                    # Blend RGB onto black background using alpha
                    image = rgb_channels * alpha_channel # Black is (0,0,0), so (0.0 - alpha_channel)*0 is implicit
            # Now image is RGB, shape [H, W, 3]
            self.all_rgbs.append(image)

        # Convert lists of numpy arrays to PyTorch tensors
        self.poses = torch.from_numpy(np.array(self.poses)).float()
        self.all_rgbs = torch.from_numpy(np.array(self.all_rgbs)).float() # [N_images, H, W, 3]

        # Calculate focal length using image width and camera horizontal field of view (camera_angle_x)
        if self.width: # Ensure width is loaded (i.e., at least one image was processed)
             self.focal = 0.5 * self.width / np.tan(0.5 * camera_angle_x)
        else:
            # Handle case where no images are loaded (e.g., empty split or all images missing)
            print("Warning: Image width not available (no images loaded/processed successfully). Focal length cannot be calculated.")
            # Optionally, set a default focal or raise an error if critical
            self.focal = None # Or some default like 555.555

        if self.transform: # Apply transform if any (usually for images, but here rgbs are already processed)
            # This dataset loads all data at init, so transform might be less common here
            # unless it's for some global adjustment or augmentation strategy not typical for NeRF.
            pass


    def __len__(self):
        """Returns the total number of samples (images) in the dataset split."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Retrieves a sample from the dataset at the given index.

        For training, this typically returns a batch of rays from the image.
        For validation/testing, it returns all rays for one image.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing:
                'rays_o' (torch.Tensor): Ray origins. Shape [H*W, 3] for train, [H, W, 3] for val/test.
                'rays_d' (torch.Tensor): Ray directions (normalized). Shape [H*W, 3] for train, [H, W, 3] for val/test.
                'target_rgb' (torch.Tensor): Target RGB values. Shape [H*W, 3] for train, [H, W, 3] for val/test.
                'pose' (torch.Tensor, optional): Camera pose. Shape [4,4]. Only for val/test.
        """
        pose = self.poses[idx] # Get camera-to-world matrix for the image
        rgb_target = self.all_rgbs[idx] # Corresponding RGB image [H, W, 3]

        # Generate all rays for the image
        # H, W, focal should be available as self.height, self.width, self.focal
        rays_o, rays_d = get_rays(self.height, self.width, self.focal, pose)

        if self.split == 'train':
            # For training, NeRF typically samples rays from across all images,
            # or if batching per image, flattens rays. Here, we return all rays for an image,
            # and the DataLoader's batch_sampler or subsequent processing would handle ray batching.
            # This implementation returns rays for a single image, flattened.
            rays_o = rays_o.reshape(-1, 3) # Shape: [H*W, 3]
            rays_d = rays_d.reshape(-1, 3) # Shape: [H*W, 3]
            rgb_target = rgb_target.reshape(-1, 3) # Shape: [H*W, 3]
            return {"rays_o": rays_o, "rays_d": rays_d, "target_rgb": rgb_target}
        else: # 'val' or 'test'
            # For validation/testing, return per-image rays and targets, and optionally the pose.
            return {"rays_o": rays_o, "rays_d": rays_d, "target_rgb": rgb_target, "pose": pose}
