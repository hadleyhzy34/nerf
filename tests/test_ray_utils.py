"""Unit tests for ray utility functions."""

import torch
import numpy as np
import pytest

# Assuming src is in PYTHONPATH or pytest is run from project root
from src.utils.ray_utils import get_rays

@pytest.fixture
def camera_params_cpu():
    """Returns basic camera parameters (H, W, focal, c2w) on CPU."""
    return {
        "H": 10,
        "W": 10,
        "focal": 15.0,
        "c2w": torch.eye(4, dtype=torch.float32) # Identity pose
    }

@pytest.fixture
def camera_params_cuda(request): # request is a built-in pytest fixture
    """Returns basic camera parameters on CUDA if available, otherwise skips the test."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for device-specific testing")
    return {
        "H": 10,
        "W": 10,
        "focal": 15.0,
        "c2w": torch.eye(4, dtype=torch.float32).cuda() # Identity pose on CUDA
    }

@pytest.mark.parametrize("params_fixture_name", ["camera_params_cpu", "camera_params_cuda"])
def test_get_rays_output_shape_and_device(params_fixture_name, request):
    """Test output shapes and device consistency of get_rays.

    Parameterized to run on both CPU and CUDA (if available).
    """
    params = request.getfixturevalue(params_fixture_name)
    H, W, focal, c2w = params["H"], params["W"], params["focal"], params["c2w"]

    rays_o, rays_d = get_rays(H, W, focal, c2w)

    # Expected shape for rays_o and rays_d is [H, W, 3] as per get_rays docstring
    assert rays_o.shape == (H, W, 3), f"Expected rays_o shape {(H, W, 3)}, got {rays_o.shape}"
    assert rays_d.shape == (H, W, 3), f"Expected rays_d shape {(H, W, 3)}, got {rays_d.shape}"

    # Check if outputs are on the same device as the input c2w matrix
    assert rays_o.device == c2w.device, "rays_o is not on the same device as c2w"
    assert rays_d.device == c2w.device, "rays_d is not on the same device as c2w"

    # Check if ray directions are normalized
    norms_d = torch.norm(rays_d, dim=-1)
    assert torch.allclose(norms_d, torch.ones_like(norms_d), atol=1e-6), "Ray directions are not normalized"


def test_get_rays_identity_pose_logic_cpu():
    """Test ray properties (origins, directions) for an identity camera pose on CPU.

    For an identity c2w matrix, camera is at origin, looking along world -Z if using
    OpenGL convention (camera's -Z is its viewing direction).
    The get_rays implementation states: "Camera looks along -Z axis in its own coordinate system".
    If c2w is identity, camera space is world space. So camera at origin, looking along world -Z.
    Ray origins should be (0,0,0).
    Center ray direction should be (0,0,-1).
    """
    H, W, focal = 4, 4, 5.0 # Small H, W for easier manual index checking
    # Identity c2w: camera at origin, aligned with world axes.
    # Camera X -> World X, Camera Y -> World Y, Camera Z -> World Z.
    # If camera looks along its -Z, it looks along World -Z.
    c2w = torch.eye(4, dtype=torch.float32)

    rays_o, rays_d = get_rays(H, W, focal, c2w)

    # For identity c2w, all ray origins should be at (0,0,0)
    assert torch.allclose(rays_o, torch.zeros_like(rays_o), atol=1e-7), \
        f"Ray origins for identity pose should be all zeros, got {rays_o}"

    # Test center ray direction
    # Pixel indices for center: (H-1)/2, (W-1)/2. If H,W are even, it's between pixels.
    # Let's take H//2, W//2 as "center-ish" pixel row/col index.
    # dirs_x = (pixels_x - W * 0.5) / focal
    # dirs_y = (pixels_y - H * 0.5) / focal
    # dirs_z = -torch.ones_like(dirs_x)
    # For pixel (W//2, H//2) (e.g. (2,2) for 4x4 image, if pixels_x/y are 0-indexed)
    # pixel_x = W//2 = 2.  (2 - 4*0.5)/focal = 0/focal = 0
    # pixel_y = H//2 = 2.  (2 - 4*0.5)/focal = 0/focal = 0
    # So, center ray in camera coords is (0, 0, -1).
    # Since c2w is identity, world direction is also (0, 0, -1).
    center_pixel_row, center_pixel_col = H // 2, W // 2
    # Note: get_rays returns H,W,3. If we use H//2, W//2, it's not the exact center if H,W are odd.
    # The formula (pixels_x - W*0.5) means the true optical center corresponds to where this is 0.
    # This happens at pixel_x = W*0.5. If W is even (e.g. 4), W*0.5=2. This is between pixel 1 and 2 (0-indexed).
    # If we take pixel indices from torch.arange(H/W), e.g., 0,1,2,3 for W=4.
    # (0 - 2)/f, (1-2)/f, (2-2)/f=0, (3-2)/f.
    # For simplicity, let's verify the principle. The center ray should be along -Z.
    # The exact index might depend on how one defines "center pixel".
    # The pixel (W*0.5 - 0.5, H*0.5 - 0.5) is the center of the top-leftmost pixel if W,H are integer.
    # The pixel whose center is closest to optical axis is (W-1)/2, (H-1)/2.
    # For H=4, W=4, this is (1.5, 1.5).
    # The meshgrid is torch.arange(H), torch.arange(W).
    # pixels_x[H//2, W//2] = W//2.  pixels_y[H//2, W//2] = H//2.
    # dir_x at [H//2, W//2] is (W//2 - W*0.5)/focal. If W is even, W//2 - W*0.5 = 0.
    # dir_y at [H//2, W//2] is (H//2 - H*0.5)/focal. If H is even, H//2 - H*0.5 = 0.
    # So for even H,W, pixel (H//2, W//2) will give (0,0,-1) direction.

    center_ray_d_actual = rays_d[H//2, W//2, :]
    expected_center_d = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
    assert torch.allclose(center_ray_d_actual, expected_center_d, atol=1e-6), \
        f"Center ray direction mismatch: got {center_ray_d_actual}, expected {expected_center_d}"

    # Test top-left corner ray direction (pixel 0,0)
    # dir_x = (0 - W*0.5)/focal = -W*0.5/focal
    # dir_y = (0 - H*0.5)/focal = -H*0.5/focal
    # dir_z = -1
    tl_ray_d_actual = rays_d[0, 0, :]
    expected_tl_d_unnormalized = torch.tensor([
        -W * 0.5 / focal,
        -H * 0.5 / focal,
        -1.0
    ], dtype=torch.float32)
    expected_tl_d_normalized = expected_tl_d_unnormalized / torch.norm(expected_tl_d_unnormalized)
    assert torch.allclose(tl_ray_d_actual, expected_tl_d_normalized, atol=1e-6), \
        f"Top-left ray direction mismatch: got {tl_ray_d_actual}, expected {expected_tl_d_normalized}"

def test_get_rays_translation_cpu():
    """Test ray origins when camera is translated."""
    H, W, focal = 4, 4, 5.0
    c2w = torch.eye(4, dtype=torch.float32)
    translation = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    c2w[:3, 3] = translation # Apply translation to camera pose

    rays_o, rays_d = get_rays(H, W, focal, c2w)

    # All ray origins should be at the camera's translated position
    expected_origins = translation.expand_as(rays_o)
    assert torch.allclose(rays_o, expected_origins, atol=1e-7), \
        f"Ray origins for translated pose should be {translation}, got average {torch.mean(rays_o, dim=(0,1))}"

    # Ray directions should be the same as identity pose because only translation changed
    identity_c2w = torch.eye(4, dtype=torch.float32)
    _, rays_d_identity = get_rays(H, W, focal, identity_c2w)
    assert torch.allclose(rays_d, rays_d_identity, atol=1e-7), \
        "Ray directions should not change with camera translation only"

# It might be useful to add a test for a simple rotation as well.
# For example, rotating camera 90 degrees around Y-axis.
# Original: looking -Z. After rot Y by +90: new -Z is along world +X. So camera looks +X.
# Center ray should be (1,0,0).
# c2w for this:
# R_y(90) = [[0,0,1],[0,1,0],[-1,0,0]]
# If camera space Z is view, then R_y(90) * [0,0,1] = [1,0,0]
# If camera space -Z is view, then R_y(90) * [0,0,-1] = [-1,0,0] (this is what our get_rays does)
# So after rotation, camera's -Z axis (viewing axis) aligns with world -X.
# Thus, center ray direction should be (-1, 0, 0).

# (The ray_utils.py get_rays returns H,W,3 not H*W,3, fixed test assertions)
