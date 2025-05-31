"""Image processing utilities, including PSNR calculation and type conversion."""

import torch
import numpy as np

def mse_to_psnr(mse):
    """Converts Mean Squared Error (MSE) to Peak Signal-to-Noise Ratio (PSNR).

    Args:
        mse (torch.Tensor or float or np.number): The Mean Squared Error value.
            If a tensor, it should be a scalar or a tensor with one element.

    Returns:
        torch.Tensor or float: The PSNR value. Returns float('inf') if MSE is 0,
        float('nan') if MSE is negative. If input was a tensor, output is a tensor
        on the same device.

    Raises:
        ValueError: If the input tensor `mse` has more than one element.
        TypeError: If the input `mse` is not a torch.Tensor, float, or np.number.
    """
    is_tensor = isinstance(mse, torch.Tensor)

    if is_tensor:
        # Ensure it's a scalar tensor or a tensor with one element
        if mse.numel() != 1:
            raise ValueError("Input mse tensor must be a scalar or have one element.")
        mse_val = mse.item() # Get scalar value for checks
        device = mse.device
    elif isinstance(mse, (float, np.number)):
        mse_val = mse
        device = None # Not applicable for float/numpy number
    else:
        raise TypeError(f"Unsupported type for mse: {type(mse)}. Must be torch.Tensor, float, or np.number.")

    if mse_val < 0: # MSE should not be negative
        # print("Warning: Negative MSE value encountered in mse_to_psnr.") # Optional warning
        return torch.tensor(float('nan'), device=device) if is_tensor else float('nan')
    if mse_val == 0:
        # Perfect match, PSNR is infinite
        return torch.tensor(float('inf'), device=device) if is_tensor else float('inf')

    # Calculate PSNR
    if is_tensor: # Prefer torch ops if input was tensor
        # Use the original tensor `mse` for calculation to preserve gradients if any,
        # though detach() is usually called before passing to mse_to_psnr for metrics.
        return -10. * torch.log10(mse)
    else:
        return -10. * np.log10(mse_val)


def to_uint8(data):
    """Converts a float tensor or numpy array from [0,1] range to uint8 [0,255].

    The input data is clipped to the [0,1] range before conversion.

    Args:
        data (torch.Tensor or np.ndarray): Input data, assumed to be in the
            nominal range [0,1]. Can be a scalar, tensor, or numpy array.

    Returns:
        np.ndarray: The converted data as a NumPy array of type uint8.
            Scalar inputs are returned as a single-element NumPy array.

    Raises:
        TypeError: If the input `data` is not a PyTorch tensor or NumPy array.
    """
    if isinstance(data, torch.Tensor):
        # Detach from graph, move to CPU, and convert to numpy
        processed_data = data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        processed_data = data
    else:
        raise TypeError(f"Input must be a PyTorch tensor or NumPy array, got {type(data)}")

    # Handle scalar input consistently by converting to array first
    if processed_data.ndim == 0:
         processed_data = np.array([processed_data])

    min_val, max_val = processed_data.min(), processed_data.max()

    # A small tolerance for floating point inaccuracies when checking range.
    if min_val < -1e-5 or max_val > 1.0 + 1e-5:
        # This warning can be noisy if inputs frequently slightly exceed the range
        # due to numerical issues. Consider removing or adjusting verbosity if needed.
        # print(f"Warning: Input data for to_uint8 was outside [0,1] (min: {min_val:.3f}, max: {max_val:.3f}) and has been clipped.")
        pass

    # Clip data to ensure it's robustly in [0,1] before scaling
    processed_data = np.clip(processed_data, 0.0, 1.0)

    return (processed_data * 255).astype(np.uint8)
