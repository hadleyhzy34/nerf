"""Unit tests for NeRF model components (PositionalEncoder, NeRF MLP, sampling)."""

import torch
import pytest
import numpy as np

# Assuming src is in PYTHONPATH or pytest is run from project root
from nerf_project.models.nerf import PositionalEncoder, NeRF, sample_pdf

@pytest.fixture
def device(request):
    """Fixture to select device (cuda if requested and available, else cpu).

    Skips tests decorated to use this fixture if CUDA is requested but not available.
    """
    if request.config.getoption("--run_cuda_tests"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            pytest.skip("CUDA not available, skipping CUDA-specific test.")
    return torch.device("cpu")

class TestPositionalEncoder:
    """Tests for the PositionalEncoder class."""

    def test_output_dims(self, device):
        """Test that output_dims attribute is calculated correctly."""
        encoder_3d_l10 = PositionalEncoder(input_dims=3, num_freqs=10).to(device)
        # Expected: 3 (original) + 3 (dims) * 2 (sin,cos) * 10 (freqs) = 3 + 60 = 63
        assert encoder_3d_l10.output_dims == 63, "Output dims incorrect for 3D input, L=10"

        encoder_2d_l5 = PositionalEncoder(input_dims=2, num_freqs=5).to(device)
        # Expected: 2 (original) + 2 (dims) * 2 (sin,cos) * 5 (freqs) = 2 + 20 = 22
        assert encoder_2d_l5.output_dims == 22, "Output dims incorrect for 2D input, L=5"

        encoder_l0 = PositionalEncoder(input_dims=3, num_freqs=0).to(device)
        # Expected: 3 (original) + 3 * 2 * 0 = 3
        assert encoder_l0.output_dims == 3, "Output dims incorrect for L=0 (should be original dims)"

    def test_forward_pass_shape(self, device):
        """Test the shape of the output tensor from the forward pass."""
        input_dims = 3
        num_freqs = 10
        encoder = PositionalEncoder(input_dims=input_dims, num_freqs=num_freqs).to(device)
        expected_out_dims = input_dims * (1 + 2 * num_freqs)

        batch_size = 5
        test_input = torch.randn(batch_size, input_dims).to(device)
        output = encoder(test_input)

        assert output.shape == (batch_size, expected_out_dims), \
            f"Expected output shape {(batch_size, expected_out_dims)}, got {output.shape}"
        assert output.device == device, "Output tensor is not on the correct device"

        # Test with more dimensions in input
        test_input_multi_dim = torch.randn(batch_size, 7, input_dims).to(device)
        output_multi_dim = encoder(test_input_multi_dim)
        assert output_multi_dim.shape == (batch_size, 7, expected_out_dims), \
            f"Expected output shape for multi-dim input {(batch_size, 7, expected_out_dims)}, got {output_multi_dim.shape}"


    def test_encoding_values_simple(self, device):
        """Test the actual encoding values for simple cases (L=0 and L=1)."""
        # Test L=0: output should be the same as input
        encoder_l0 = PositionalEncoder(input_dims=1, num_freqs=0).to(device)
        test_input_l0 = torch.tensor([[1.5], [-0.5]]).to(device)
        output_l0 = encoder_l0(test_input_l0)
        assert encoder_l0.output_dims == 1, "Output dims for L=0 should be input_dims"
        assert torch.allclose(output_l0, test_input_l0), "Output for L=0 should be identical to input"

        # Test L=1 (one frequency band)
        # If log_sampling=True (default), freq_bands = [2^0] = [1.0]
        # If log_sampling=False, freq_bands for num_freqs=1 is linspace(1, 2^0, 1) = [1.0]
        encoder_l1 = PositionalEncoder(input_dims=1, num_freqs=1, log_sampling=True).to(device)
        assert encoder_l1.output_dims == 1 + 1 * 2 * 1, "Output dims for L=1 incorrect"

        # Input p = pi / 2, freq = 1.0
        # Output should be [p, sin(p*1), cos(p*1)]
        p_val = np.pi / 2.0
        p_tensor = torch.tensor([[p_val]]).to(device)
        output_l1 = encoder_l1(p_tensor)
        expected_l1 = torch.tensor([[p_val, np.sin(p_val * 1.0), np.cos(p_val * 1.0)]]).to(device)
        assert torch.allclose(output_l1, expected_l1, atol=1e-7), \
            f"Encoding for L=1, p=pi/2 failed. Got {output_l1}, expected {expected_l1}"

        # Input 0, freq = 1.0
        # Output should be [0, sin(0), cos(0)] = [0, 0, 1]
        zero_input = torch.zeros(1, 1, device=device)
        output_zero = encoder_l1(zero_input)
        expected_zero = torch.tensor([[0.0, 0.0, 1.0]]).to(device)
        assert torch.allclose(output_zero, expected_zero, atol=1e-7), \
            f"Encoding for L=1, p=0 failed. Got {output_zero}, expected {expected_zero}"


class TestNeRFModel:
    """Tests for the NeRF MLP class."""

    @pytest.mark.parametrize("use_viewdirs", [True, False])
    def test_nerf_forward_pass_shape(self, use_viewdirs, device):
        """Test the NeRF model's forward pass output shape and device."""
        input_ch_pts = 63  # Example: 3D points with L=10
        input_ch_views = 27 if use_viewdirs else 0 # Example: 3D views with L=4

        # Using smaller D and W for faster test execution
        nerf_model = NeRF(D=4, W=64,
                          input_ch_pts=input_ch_pts,
                          input_ch_views=input_ch_views,
                          use_viewdirs=use_viewdirs,
                          skips=[2]).to(device) # Skip after layer 2 (0-indexed)

        num_rays, num_samples_per_ray = 5, 10 # Simulating batched samples
        total_points = num_rays * num_samples_per_ray

        pts_encoded = torch.randn(total_points, input_ch_pts).to(device)
        views_encoded = None
        if use_viewdirs:
            views_encoded = torch.randn(total_points, input_ch_views).to(device)

        output = nerf_model(pts_encoded, views_encoded)

        # Output should be [total_points, 4] (RGB + sigma)
        assert output.shape == (total_points, 4), \
            f"Expected output shape {(total_points, 4)}, got {output.shape}"
        assert output.device == device, "Output tensor is not on the correct device"

    def test_nerf_skip_connection_layer_structure(self, device):
        """Test the layer dimensions when skip connections are used."""
        D_test = 8
        W_test = 256
        input_ch_pts_test = 63
        skips_test = [4] # Skip connection after layer 4 (0-indexed)

        model = NeRF(D=D_test, W=W_test,
                     input_ch_pts=input_ch_pts_test,
                     skips=skips_test,
                     use_viewdirs=False # Viewdirs don't affect pts_linears structure
                    ).to(device)

        # Layer 0 (pts_linears[0])
        assert model.pts_linears[0].in_features == input_ch_pts_test
        assert model.pts_linears[0].out_features == W_test

        # The layer that receives the concatenated skip connection input.
        # In NeRF's __init__, pts_linears are created:
        #   pts_linears[0] is Linear(input_ch_pts, W)
        #   Loop `i` from 0 to D-2 creates `pts_linears[i+1]`.
        #   If `i` (from loop) is in `skips`, then `pts_linears[i+1]` gets `W + input_ch_pts`.
        # In forward pass:
        #   Loop `i` from 0 to D-1 (layer index).
        #   If `i` (layer index) is in `skips`, output of `pts_linears[i]` is concatenated.
        #   This concatenated output is input to `pts_linears[i+1]`.
        # So, `pts_linears[skips[0] + 1]` should have `W + input_ch_pts` as in_features.

        target_layer_idx_for_skip_input = skips_test[0] + 1 # This layer processes concatenated input

        for k in range(1, D_test): # Iterate through layers 1 to D-1
            layer = model.pts_linears[k]
            if k == target_layer_idx_for_skip_input:
                assert layer.in_features == W_test + input_ch_pts_test, \
                    f"Layer {k} (skip input layer) in_features should be {W_test + input_ch_pts_test}, got {layer.in_features}"
            else:
                assert layer.in_features == W_test, \
                    f"Layer {k} in_features should be {W_test}, got {layer.in_features}"
            assert layer.out_features == W_test, \
                f"Layer {k} out_features should be {W_test}, got {layer.out_features}"


class TestSampling:
    """Tests for sampling functions like sample_pdf."""

    def test_sample_pdf_output_shape(self, device):
        """Test the output shape of sample_pdf."""
        N_rays = 5
        N_coarse_bins_for_pdf = 63 # Number of intervals, so weights are N_coarse-1
        N_importance = 128        # Number of fine samples to draw

        # `bins` are midpoints of intervals, so N_coarse_bins_for_pdf (or N_samples_coarse-1)
        bins = torch.rand(N_rays, N_coarse_bins_for_pdf, device=device)
        weights = torch.rand(N_rays, N_coarse_bins_for_pdf, device=device) # Same number of weights as bins

        samples = sample_pdf(bins, weights, N_importance, det=False) # Stochastic sampling
        assert samples.shape == (N_rays, N_importance), \
            f"Expected samples shape {(N_rays, N_importance)}, got {samples.shape}"
        assert samples.device == device, "Output samples are not on the correct device"

    def test_sample_pdf_deterministic_and_bounds(self, device):
        """Test deterministic sampling properties and that samples are within bin bounds."""
        N_rays = 2
        # Bins are midpoints of intervals. For 5 bins, values could be e.g., 0.1, 0.3, 0.5, 0.7, 0.9
        # These define 5 intervals.
        bins_data = torch.tensor([[0.1, 0.3, 0.5, 0.7, 0.9],    # Ray 1 bins
                                  [0.2, 0.4, 0.6, 0.8, 1.0]],   # Ray 2 bins
                                 device=device, dtype=torch.float32)

        # Uniform weights means PDF is uniform over the bins provided.
        weights_data = torch.ones_like(bins_data, device=device)

        N_importance = 5 # Sample 5 points
        samples_det = sample_pdf(bins_data, weights_data, N_importance, det=True) # Deterministic

        # Check that samples are within the range of the provided bins (min and max of midpoints)
        # A sample can, by interpolation, go slightly outside the midpoint if it's at the edge of CDF.
        # The actual range is defined by the implicit edges around these midpoints.
        # For a piece-wise constant PDF defined by midpoints, the samples should fall
        # roughly within the span of these midpoints. More precisely, they are interpolated
        # between bin values. So, they should be bounded by min(bins) and max(bins).
        assert torch.all(samples_det >= bins_data.min(dim=-1, keepdim=True).values - 1e-6), \
            "Deterministic samples are below the minimum bin value."
        assert torch.all(samples_det <= bins_data.max(dim=-1, keepdim=True).values + 1e-6), \
            "Deterministic samples are above the maximum bin value."

        # For deterministic sampling with uniform weights over uniform bins, samples should be sorted.
        # (This might not hold for non-uniform bins or weights, but for this simple case it should)
        if torch.allclose(torch.diff(bins_data[0]), torch.diff(bins_data[0])[0]): # If bins are uniform
            sorted_samples, _ = torch.sort(samples_det, dim=-1)
            assert torch.allclose(samples_det, sorted_samples, atol=1e-7), \
                "Deterministic samples from uniform PDF over uniform bins are not sorted as expected."

    def test_sample_pdf_empty_weights(self, device):
        """Test sample_pdf behavior when all weights are zero (or near zero after epsilon)."""
        N_rays = 2
        N_coarse_bins_for_pdf = 5
        N_importance = 10

        bins = torch.linspace(0.1, 0.9, steps=N_coarse_bins_for_pdf, device=device).unsqueeze(0).expand(N_rays, -1)
        weights = torch.zeros_like(bins, device=device) # All weights are zero

        samples = sample_pdf(bins, weights, N_importance, det=False) # Stochastic

        assert samples.shape == (N_rays, N_importance)
        # With zero weights (plus epsilon), PDF becomes ~uniform. Samples should still be within bin range.
        assert torch.all(samples >= bins.min(dim=-1, keepdim=True).values - 1e-6)
        assert torch.all(samples <= bins.max(dim=-1, keepdim=True).values + 1e-6)

        samples_det = sample_pdf(bins, weights, N_importance, det=True) # Deterministic
        assert samples_det.shape == (N_rays, N_importance)
        assert torch.all(samples_det >= bins.min(dim=-1, keepdim=True).values - 1e-6)
        assert torch.all(samples_det <= bins.max(dim=-1, keepdim=True).values + 1e-6)
```
