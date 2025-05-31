"""Unit tests for image utility functions."""

import torch
import numpy as np
import pytest

# Assuming src is in PYTHONPATH or pytest is run from project root
from nerf_project.utils.image_utils import mse_to_psnr, to_uint8

def test_mse_to_psnr_scalar():
    """Test mse_to_psnr with scalar inputs."""
    assert mse_to_psnr(0.0) == float('inf'), "PSNR for MSE=0 should be inf"
    assert mse_to_psnr(1.0) == 0.0, "PSNR for MSE=1 should be 0"
    assert np.isclose(mse_to_psnr(0.1), 10.0), "PSNR for MSE=0.1 should be 10"
    assert np.isclose(mse_to_psnr(0.01), 20.0), "PSNR for MSE=0.01 should be 20"
    assert np.isnan(mse_to_psnr(-1.0)), "PSNR for negative MSE should be NaN"

def test_mse_to_psnr_tensor():
    """Test mse_to_psnr with torch.Tensor inputs."""
    assert mse_to_psnr(torch.tensor(0.0)) == float('inf'), "PSNR for Tensor MSE=0 should be inf"
    assert mse_to_psnr(torch.tensor(1.0)).item() == 0.0, "PSNR for Tensor MSE=1 should be 0"
    assert torch.isclose(mse_to_psnr(torch.tensor(0.1)), torch.tensor(10.0)), "PSNR for Tensor MSE=0.1 should be 10"
    assert torch.isclose(mse_to_psnr(torch.tensor(0.01)), torch.tensor(20.0)), "PSNR for Tensor MSE=0.01 should be 20"
    assert torch.isnan(mse_to_psnr(torch.tensor(-1.0))), "PSNR for negative Tensor MSE should be NaN"

def test_to_uint8_numpy():
    """Test to_uint8 with NumPy array inputs."""
    data_np = np.array([[0.0, 0.5, 1.0], [0.2, 0.7, 0.999]])
    # Expected: 0.5*255 = 127.5 -> 127; 0.2*255 = 51; 0.7*255 = 178.5 -> 178; 0.999*255 = 254.745 -> 254
    expected_np = np.array([[0, 127, 255], [51, 178, 254]], dtype=np.uint8)
    result_np = to_uint8(data_np)
    assert np.array_equal(result_np, expected_np), "NumPy array conversion failed"
    assert result_np.dtype == np.uint8, "NumPy result dtype is not uint8"

    data_clip_np = np.array([-0.5, 0.5, 1.5]) # Test clipping
    expected_clip_np = np.array([0, 127, 255], dtype=np.uint8)
    assert np.array_equal(to_uint8(data_clip_np), expected_clip_np), "NumPy array clipping failed"

    # Test scalar numpy input
    scalar_np = np.array(0.6)
    expected_scalar_np = np.array([153], dtype=np.uint8) # 0.6 * 255 = 153
    assert np.array_equal(to_uint8(scalar_np), expected_scalar_np), "Scalar NumPy array conversion failed"


def test_to_uint8_tensor():
    """Test to_uint8 with torch.Tensor inputs."""
    data_tensor = torch.tensor([[0.0, 0.5, 1.0], [0.2, 0.7, 0.999]])
    expected_tensor = np.array([[0, 127, 255], [51, 178, 254]], dtype=np.uint8)
    result_tensor = to_uint8(data_tensor)
    assert np.array_equal(result_tensor, expected_tensor), "Tensor conversion failed"
    assert result_tensor.dtype == np.uint8, "Tensor result dtype is not uint8"

    data_clip_tensor = torch.tensor([-0.5, 0.5, 1.5]) # Test clipping
    expected_clip_tensor = np.array([0, 127, 255], dtype=np.uint8)
    assert np.array_equal(to_uint8(data_clip_tensor), expected_clip_tensor), "Tensor clipping failed"

    scalar_tensor = torch.tensor(0.6) # Test scalar tensor
    expected_scalar_tensor = np.array([153], dtype=np.uint8) # 0.6 * 255 = 153
    assert np.array_equal(to_uint8(scalar_tensor), expected_scalar_tensor), "Scalar tensor conversion failed"


def test_to_uint8_type_error():
    """Test to_uint8 raises TypeError for unsupported input types."""
    with pytest.raises(TypeError):
        to_uint8([0.1, 0.2]) # Should raise TypeError for list input

    with pytest.raises(TypeError):
        to_uint8("not a tensor or array") # Should raise TypeError for string input

# Additional tests for mse_to_psnr for edge cases or specific device if needed
def test_mse_to_psnr_tensor_on_device():
    """Test mse_to_psnr with a tensor on a specific device if CUDA is available."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        mse_tensor_cuda = torch.tensor(0.01, device=device)
        psnr_tensor_cuda = mse_to_psnr(mse_tensor_cuda)
        assert torch.isclose(psnr_tensor_cuda, torch.tensor(20.0, device=device)), "PSNR calculation on CUDA failed"
        assert psnr_tensor_cuda.device == device, "PSNR tensor not on correct CUDA device"
    else:
        pytest.skip("CUDA not available for device-specific tensor test")

def test_mse_to_psnr_value_error():
    """Test mse_to_psnr raises ValueError for non-scalar tensor."""
    with pytest.raises(ValueError):
        mse_to_psnr(torch.tensor([0.1, 0.2]))
