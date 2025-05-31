import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PositionalEncoder(nn.Module):
    def __init__(self, input_dims, num_freqs, log_sampling=True):
        """
        Initializes the PositionalEncoder.
        Args:
            input_dims (int): Dimensionality of the input.
            num_freqs (int): Number of frequencies L.
            log_sampling (bool): Whether to sample frequencies in log scale.
        """
        super().__init__()
        self.input_dims = input_dims
        self.num_freqs = num_freqs
        self.log_sampling = log_sampling

        if self.log_sampling:
            self.freq_bands = 2.**torch.linspace(0., num_freqs - 1, num_freqs)
        else:
            freq_bands = torch.linspace(1., 2.**(num_freqs - 1), num_freqs)

        self.register_buffer('freq_bands', freq_bands if 'freq_bands' in locals() else 2.**torch.linspace(0., num_freqs - 1, num_freqs))
        # Output dimension: original input_dims + input_dims * 2 * num_freqs for sin and cos
        self.output_dims = self.input_dims + self.input_dims * 2 * self.num_freqs

    def forward(self, x):
        """
        Applies positional encoding to the input tensor.
        Args:
            x (torch.Tensor): Input tensor of shape [N, ..., input_dims].
        Returns:
            torch.Tensor: Encoded tensor of shape [N, ..., output_dims].
        """
        outputs = [x] # Include the original input
        for freq in self.freq_bands:
            outputs.append(torch.sin(x * freq))
            outputs.append(torch.cos(x * freq))

        return torch.cat(outputs, dim=-1)

if __name__ == '__main__':
    # Example Usage
    input_dim = 3
    L = 10
    encoder = PositionalEncoder(input_dim, L)
    print(f"Output dimensions: {encoder.output_dims}") # Expected: 3 + 3*2*10 = 63

    # Test with a sample tensor
    # Batch of 2, sequence of 5 points, each 3D
    test_input = torch.rand(2, 5, input_dim)
    encoded_output = encoder(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Encoded output shape: {encoded_output.shape}") # Expected: [2, 5, 63]

    # Test with a single point
    test_single_input = torch.rand(input_dim)
    encoded_single_output = encoder(test_single_input)
    print(f"Single input shape: {test_single_input.shape}")
    print(f"Encoded single output shape: {encoded_single_output.shape}") # Expected: [63]

    # Test with batch of points (N, D)
    test_batch_points = torch.rand(100, input_dim)
    encoded_batch_points = encoder(test_batch_points)
    print(f"Batch points input shape: {test_batch_points.shape}")
    print(f"Encoded batch points output shape: {encoded_batch_points.shape}") # Expected: [100, 63]

    # Test different input_dims and L
    encoder_2d_L4 = PositionalEncoder(input_dims=2, num_freqs=4)
    print(f"2D L=4 Output dimensions: {encoder_2d_L4.output_dims}") # Expected: 2 + 2*2*4 = 18
    test_input_2d = torch.rand(3, 2)
    encoded_output_2d = encoder_2d_L4(test_input_2d)
    print(f"2D L=4 Input shape: {test_input_2d.shape}")
    print(f"2D L=4 Encoded output shape: {encoded_output_2d.shape}") # Expected: [3, 18]

    # Test with log_sampling=False
    encoder_linear_sampling = PositionalEncoder(input_dim, L, log_sampling=False)
    print(f"Output dimensions (linear): {encoder_linear_sampling.output_dims}")
    encoded_output_linear = encoder_linear_sampling(test_input)
    print(f"Encoded output shape (linear): {encoded_output_linear.shape}")

    assert encoded_output.shape[-1] == encoder.output_dims
    assert encoded_single_output.shape[-1] == encoder.output_dims
    assert encoded_batch_points.shape[-1] == encoder.output_dims
    assert encoded_output_2d.shape[-1] == encoder_2d_L4.output_dims
    assert encoded_output_linear.shape[-1] == encoder_linear_sampling.output_dims
    print("PositionalEncoder tests passed.")


class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch_pts=63, input_ch_views=27, skips=[4], use_viewdirs=True):
        """
        Initializes the NeRF model.
        Args:
            D (int): Number of linear layers in the main network.
            W (int): Number of hidden units per layer.
            input_ch_pts (int): Number of channels for encoded input 3D coordinates.
            input_ch_views (int): Number of channels for encoded input viewing directions.
            skips (list of int): List of layer indices where residual connections should be added.
            use_viewdirs (bool): Whether to use viewing directions.
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch_pts = input_ch_pts
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        # Layers for processing spatial coordinates (pts_linears)
        self.pts_linears = nn.ModuleList()
        self.pts_linears.append(nn.Linear(input_ch_pts, W))
        for i in range(D - 1):
            if i in self.skips:
                self.pts_linears.append(nn.Linear(W + input_ch_pts, W))
            else:
                self.pts_linears.append(nn.Linear(W, W))

        # Layers for output
        if use_viewdirs:
            self.sigma_linear = nn.Linear(W, 1)  # For sigma (density)
            self.feature_linear = nn.Linear(W, W) # To get feature vector

            # Layers for predicting RGB using features and view directions
            self.rgb_layers = nn.Sequential(
                nn.Linear(W + input_ch_views, W // 2),
                nn.ReLU(),
                nn.Linear(W // 2, 3)
            )
        else:
            # If not using view directions, predict RGB and sigma directly from the main network's output
            self.output_linear = nn.Linear(W, 4)  # Output: RGB (3) + Sigma (1)

    def forward(self, x_pts_encoded, x_views_encoded=None):
        """
        Forward pass of the NeRF model.
        Args:
            x_pts_encoded (torch.Tensor): Encoded 3D coordinates. Shape: [N, ..., input_ch_pts].
            x_views_encoded (torch.Tensor, optional): Encoded viewing directions. Shape: [N, ..., input_ch_views].
                                                     Required if use_viewdirs is True.
        Returns:
            torch.Tensor: Output tensor containing RGB (3 channels) and sigma (1 channel). Shape: [N, ..., 4].
        """
        h = x_pts_encoded
        for i, layer in enumerate(self.pts_linears):
            h = layer(h)
            h = nn.functional.relu(h)
            if i in self.skips and i < len(self.pts_linears) -1 : # Make sure skip connection is not applied after last layer of pts_linears before feature_linear/output_linear
                h = torch.cat([x_pts_encoded, h], -1)

        if self.use_viewdirs:
            if x_views_encoded is None:
                raise ValueError("View directions (x_views_encoded) must be provided when use_viewdirs is True.")

            sigma = self.sigma_linear(h)
            features = self.feature_linear(h)

            combined = torch.cat([features, x_views_encoded], -1)
            rgb = self.rgb_layers(combined)
        else:
            outputs = self.output_linear(h)
            rgb = outputs[..., :3]
            sigma = outputs[..., 3:4] # Keep last dim as 1 for sigma

        return torch.cat([rgb, sigma], -1)


if __name__ == '__main__':
    # Example Usage for NeRF model
    # Positional Encoders (already tested above, but let's make instances for NeRF)
    L_pts = 10
    L_views = 4
    pts_encoder = PositionalEncoder(3, L_pts)
    views_encoder = PositionalEncoder(3, L_views) # Assuming 3D view directions (unit vectors)

    input_ch_pts = pts_encoder.output_dims # 3 + 3*2*10 = 63
    input_ch_views = views_encoder.output_dims # 3 + 3*2*4 = 27

    # Test NeRF with view directions
    nerf_model_vd = NeRF(input_ch_pts=input_ch_pts, input_ch_views=input_ch_views, use_viewdirs=True)

    # Create dummy input
    batch_size = 4
    num_points = 1024 # e.g. rays in a batch
    raw_pts = torch.rand(batch_size, num_points, 3)
    raw_views = torch.rand(batch_size, num_points, 3)
    raw_views = raw_views / torch.norm(raw_views, dim=-1, keepdim=True) # Normalize view directions

    encoded_pts = pts_encoder(raw_pts)
    encoded_views = views_encoder(raw_views)

    print(f"\n--- NeRF with view directions ---")
    print(f"Raw pts shape: {raw_pts.shape}")
    print(f"Encoded pts shape: {encoded_pts.shape}") # Expected: [B, N, 63]
    print(f"Raw views shape: {raw_views.shape}")
    print(f"Encoded views shape: {encoded_views.shape}") # Expected: [B, N, 27]

    output_vd = nerf_model_vd(encoded_pts, encoded_views)
    print(f"Output shape (with viewdirs): {output_vd.shape}") # Expected: [B, N, 4]
    assert output_vd.shape == (batch_size, num_points, 4)

    # Test NeRF without view directions
    nerf_model_no_vd = NeRF(input_ch_pts=input_ch_pts, use_viewdirs=False)
    print(f"\n--- NeRF without view directions ---")
    output_no_vd = nerf_model_no_vd(encoded_pts) # x_views_encoded is None
    print(f"Output shape (without viewdirs): {output_no_vd.shape}") # Expected: [B, N, 4]
    assert output_no_vd.shape == (batch_size, num_points, 4)

    # Test skip connection logic carefully
    # A skip connection happens AT layer i (meaning, after layer i-1's output, before layer i's input)
    # The current pts_linears definition:
    # self.pts_linears.append(nn.Linear(input_ch_pts, W)) -> layer 0
    # for i in range(D - 1): -> i from 0 to D-2
    #   if i in self.skips: self.pts_linears.append(nn.Linear(W + input_ch_pts, W)) -> this is layer i+1
    #   else: self.pts_linears.append(nn.Linear(W, W)) -> this is layer i+1
    # So, if skips=[4], it means layer 5 (index 4 in the loop, which is pts_linears[4+1]) gets concatenated input.
    # Forward pass:
    # for i, layer in enumerate(self.pts_linears):
    #   h = layer(h)
    #   ...
    #   if i in self.skips: h = torch.cat([x_pts_encoded, h], -1)
    # This means if i=4 is in skips, after layer 4 (pts_linears[4]) computes, its output is concatenated with original x_pts_encoded.
    # This concatenated result becomes input for layer 5 (pts_linears[5]).
    # So, pts_linears[5] should be nn.Linear(W + input_ch_pts, W).
    # Let's trace for D=8, skips=[4]
    # pts_linears[0]: Linear(input_ch_pts, W)
    # Loop i = 0..6:
    # i=0: pts_linears[1] = Linear(W,W)
    # i=1: pts_linears[2] = Linear(W,W)
    # i=2: pts_linears[3] = Linear(W,W)
    # i=3: pts_linears[4] = Linear(W,W) (This is the layer *before* the one that takes skip connection)
    # i=4: (i=4 is in skips) pts_linears[5] = Linear(W + input_ch_pts, W)
    # i=5: pts_linears[6] = Linear(W,W)
    # i=6: pts_linears[7] = Linear(W,W)
    # Total D=8 layers (indices 0 to 7).
    # Forward pass:
    # h = x_pts_encoded
    # i=0: h = relu(pts_linears[0](h))
    # i=1: h = relu(pts_linears[1](h))
    # i=2: h = relu(pts_linears[2](h))
    # i=3: h = relu(pts_linears[3](h))
    # i=4: h = relu(pts_linears[4](h)). Now i=4 is in skips. h = cat(x_pts_encoded, h). This h is input to pts_linears[5].
    # This matches the original NeRF paper: "concatenate with the original input".
    # The condition `i < len(self.pts_linears) -1` in forward for skip connection is important.
    # A skip connection should not be applied after the *final* layer of pts_linears,
    # because that `h` is then used for sigma_linear or output_linear.
    # The loop for pts_linears goes up to D-1.
    # If D-1 is in skips, then after the last layer, it would concat. This is usually not intended.
    # Original NeRF: "input to the 5th layer (index 4) is the concatenation of the output of the 4th layer and the original input"
    # This means after layer index 3's output, we concat and pass to layer index 4.
    # My current code: if 4 is in skips, after layer `pts_linears[4]` output, we concat.
    # This means `pts_linears[5]` takes the concatenated input.
    # The original paper's diagram shows skip from input to layer 4.
    # If layers are 0..8 (9 layers total, D=9 in paper for pts path), skip is to input of layer 4.
    # My D is number of layers. If D=8, layers 0..7.
    # If skips=[4], it means input to layer 5 (index 4 in the loop creating layers 1 to D-1) is concatenated.
    # Let's re-check the `pts_linears` construction logic.
    # `self.pts_linears.append(nn.Linear(input_ch_pts, W))` -> This is layer 0.
    # Loop `for i in range(D - 1)`: (i goes from 0 to D-2)
    #   `self.pts_linears.append(...)` -> This appends layer 1 to layer D-1.
    #   Layer `j` (1-indexed in this loop description, so layer `i+1` in terms of list index)
    #   If `i` (0 to D-2) is in `skips`, then layer `i+1` gets `W + input_ch_pts` as input dim.
    #   Example: D=8, skips=[4].
    #   Layer 0: Linear(input_ch_pts, W)
    #   Loop i=0..6:
    #   i=0: layer 1: Linear(W,W)
    #   i=1: layer 2: Linear(W,W)
    #   i=2: layer 3: Linear(W,W)
    #   i=3: layer 4: Linear(W,W)
    #   i=4: (4 is in skips) layer 5: Linear(W+input_ch_pts, W)
    #   i=5: layer 6: Linear(W,W)
    #   i=6: layer 7: Linear(W,W)
    # This seems correct: layer 5 (index 5 in `self.pts_linears`) is the one that processes the concatenated features.
    # The `forward` pass:
    # `h = x_pts_encoded`
    # `for i, layer in enumerate(self.pts_linears):` (i goes from 0 to D-1, matching layer index)
    #   `h = layer(h)`
    #   `h = nn.functional.relu(h)`
    #   `if i in self.skips:` (e.g. i=4 is in skips)
    #     `h = torch.cat([x_pts_encoded, h], -1)`
    # This means the output of layer 4 (index 4) is concatenated with `x_pts_encoded`.
    # This concatenated `h` then becomes the input to layer 5 (index 5). This is correct.
    # The condition `i < len(self.pts_linears) -1` for skip connection application is good.
    # If `skips` contained `D-1`, the output of the last layer would be concatenated, then immediately used by `sigma_linear` etc.
    # This would change the input dimension to `sigma_linear` if `D-1` was a skip index.
    # The current code is fine. `skips` should contain indices from `0` to `D-2`.
    # (If `skips` contains `k`, then layer `k+1` is the one that receives the concatenated input).
    # The original paper mentions D=8 layers for the main MLP. This means indices 0 through 7.
    # And skip connection at layer 4. This means layer with index 4 (the 5th layer) receives concatenated input.
    # So if my D=8, then skips should be [3] for layer 4 to get concatenated input.
    # (After layer 3's output, concat, then feed to layer 4).
    # Let's adjust the interpretation: `skips` are the layers *whose output* gets concatenated.
    # No, the standard is: `skips` are layer indices `k` such that layer `k+1` takes `[output_of_k, original_input]`.
    # My current code: `if i in self.skips: h = torch.cat([x_pts_encoded, h], -1)`. `h` here is output of layer `i`.
    # So layer `i+1` receives this concatenated `h`.
    # So if `skips = [4]`, layer 5 receives concatenated input. This matches "input to the 5th layer".
    # The number of layers D counts the first layer too. So D=8 means layers 0,1,2,3,4,5,6,7.
    # This means my `skips=[4]` is consistent with original NeRF (input to 5th layer, which is index 4).
    # The `range(D-1)` in `__init__` creates `D-1` layers. Plus the first one, makes `D` layers.
    # `pts_linears` will have length `D`. Indices `0` to `D-1`.
    # If `skips=[4]`, and `D=8`.
    # In `__init__`: `i` goes from `0` to `6`.
    # `layer_idx_in_list = i + 1`.
    # If `i=4` is in `skips`, then `pts_linears[5]` is `Linear(W+input_ch_pts, W)`.
    # In `forward`: `i` goes `0` to `D-1` (i.e. `0..7`).
    # If `i=4` (which is `pts_linears[4]`), its output `h` is taken. `h = cat(x_pts, h)`.
    # This `h` is fed to `pts_linears[5]`. This is correct.
    print("NeRF model tests seem okay.")


# Hierarchical sampling helper
def sample_pdf(bins, weights, N_samples, det=False):
    """
    Sample @N_samples more points from @bins with distribution defined by @weights.
    Args:
        bins (torch.Tensor): [N_rays, N_samples_coarse - 1]. Midpoints of coarse samples.
        weights (torch.Tensor): [N_rays, N_samples_coarse - 1]. Weights for each bin.
        N_samples (int): Number of samples to draw (N_importance).
        det (bool): If True, deterministic sampling (linspace in CDF), else stochastic.
    Returns:
        samples (torch.Tensor): [N_rays, N_samples]. Fine samples.
    """
    # Add a small epsilon to weights to prevent NaN issues if all weights are zero.
    weights = weights + 1e-5  # [N_rays, N_samples_coarse - 1]
    pdf = weights / torch.sum(weights, -1, keepdim=True)  # [N_rays, N_samples_coarse - 1]
    cdf = torch.cumsum(pdf, -1)  # [N_rays, N_samples_coarse - 1]
    # Prepend zeros to CDF to have a starting point for searchsorted.
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # [N_rays, N_samples_coarse]

    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=bins.device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples]) # [N_rays, N_samples]
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=bins.device) # [N_rays, N_samples]

    u = u.contiguous() # Ensure contiguity for searchsorted

    # Find indices where `u` would be inserted into `cdf` to maintain order.
    # `right=True` means that if u[i] is equal to cdf[j], then inds[i] = j+1
    inds = torch.searchsorted(cdf, u, right=True) # [N_rays, N_samples]

    # Calculate lower and upper bin indices for interpolation
    below = torch.max(torch.zeros_like(inds - 1), inds - 1) # [N_rays, N_samples]
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds) # [N_rays, N_samples]

    # Stack for gathering: shape [N_rays, N_samples, 2]
    inds_g = torch.stack([below, above], -1)

    # Gather CDF values and bin values corresponding to these indices
    # cdf_g will have shape [N_rays, N_samples, 2]
    # inds_g.view(-1,2) has shape [N_rays*N_samples, 2]
    # cdf has shape [N_rays, N_samples_coarse]
    # We want to gather along the last dimension of cdf.
    # Matched_cdf_values = cdf[..., inds_g] does not work due to shape mismatch
    # Need to expand cdf or use gather carefully

    # For gather, input tensor and index tensor must have same number of dimensions,
    # except for the dimension specified by `dim`.
    # cdf: [N_rays, N_samples_coarse]
    # inds_g: [N_rays, N_samples, 2]
    # We want to select N_samples_coarse values for each of the N_rays.
    # The indices in inds_g are for the N_samples_coarse dimension.

    # Reshape inds_g to [N_rays, N_samples * 2] then gather, then reshape back.
    # Or, more simply, expand cdf and bins to match inds_g's N_samples dimension.
    # cdf_expanded = cdf.unsqueeze(1).expand(cdf.shape[0], N_samples, cdf.shape[1])
    # cdf_g = torch.gather(cdf_expanded, 2, inds_g)

    # Alternative gather:
    # The gather is tricky. Let's use view and then gather.
    # N_rays, N_samples_coarse = cdf.shape
    # _, N_fine_samples, _ = inds_g.shape
    # cdf_g = torch.gather(cdf.unsqueeze(1).repeat(1,N_fine_samples,1), 2, inds_g)
    # bins_g = torch.gather(bins.unsqueeze(1).repeat(1,N_fine_samples,1), 2, inds_g)

    # A simpler approach for gather:
    # Flatten inds_g for batch gather, then reshape.
    # This selects values from cdf for each ray based on the indices in inds_g.
    # cdf_g = torch.gather(cdf, -1, inds_g.view(inds_g.shape[0], -1)).view(inds_g.shape)
    # This is not quite right. Need to ensure that for each ray, we are indexing into its own CDF.
    # The `torch.gather` documentation:
    # out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
    # out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2 (our case, dim=-1)

    # Let's make sure the shapes are compatible for batch gather
    # cdf is [N_rays, N_samples_coarse]
    # inds_g is [N_rays, N_samples, 2]
    # Goal: cdf_g is [N_rays, N_samples, 2] where cdf_g[r, s, 0] = cdf[r, inds_g[r,s,0]]

    # We can iterate or use a more advanced gather.
    # For simplicity, let's prepare dummy indices for the batch dimension of cdf.
    batch_size = cdf.shape[0]
    # This will create indices for the first dimension
    idx_dim0 = torch.arange(batch_size, device=bins.device).view(-1, 1, 1)
    # Repeat for N_samples and for the 2 (below/above)
    idx_dim0 = idx_dim0.expand(-1, N_samples, 2)

    # Now we can use cdf[idx_dim0, inds_g] if this advanced indexing is supported as such
    # PyTorch advanced indexing: x[idx_dim0, inds_g]
    # cdf_g = cdf[idx_dim0, inds_g] # This works!
    # bins_g = bins[idx_dim0, inds_g]

    # Or, using gather:
    # To use gather, cdf needs to be [N_rays, N_samples_coarse]
    # inds_g is [N_rays, N_samples, 2].
    # We need to gather along dim 1 (the N_samples_coarse dimension).
    # cdf.gather(1, index_tensor_of_same_dim_as_cdf)
    # So, inds_g needs to be reshaped or used carefully.

    # Let's use the example from NeRF Pytorch code which is usually:
    # cdf_g = torch.gather(cdf, dim=-1, index=inds_g.view(batch_size, -1))
    # cdf_g = cdf_g.view(batch_size, N_samples, 2)
    # This assumes cdf is [batch_size, num_bins] and inds_g is [batch_size, N_fine, 2]
    # This seems fine:
    cdf_g = torch.gather(cdf, dim=1, index=inds_g.view(batch_size, -1)).view(batch_size, N_samples, 2)
    bins_g = torch.gather(bins, dim=1, index=inds_g.view(batch_size, -1)).view(batch_size, N_samples, 2)


    # Linearly interpolate to get new t-samples
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    # Avoid division by zero if cdf_g[...,1] == cdf_g[...,0] (e.g. if a PDF bin is zero)
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom # [N_rays, N_samples]

    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0]) # [N_rays, N_samples]
    return samples


def render_rays(
    nerf_model: NeRF,
    embed_fn_pts, # PositionalEncoder for points
    embed_fn_views, # PositionalEncoder for view directions
    rays_o: torch.Tensor, # [N_rays, 3]
    rays_d: torch.Tensor, # [N_rays, 3]
    near: float,
    far: float,
    N_samples: int, # Number of coarse samples per ray
    rand: bool = False, # If True, jitter samples
    lindisp: bool = False, # If True, sample linearly in disparity, else linearly in depth
    use_viewdirs: bool = True,
    white_bkgd: bool = False,
    z_vals_override: torch.Tensor = None # [N_rays, N_override_samples]
):
    """
    Render rays using the NeRF model.
    Args:
        nerf_model: The NeRF nn.Module.
        embed_fn_pts: Positional encoding function for 3D points.
        embed_fn_views: Positional encoding function for view directions.
        rays_o: Origins of rays [N_rays, 3].
        rays_d: Directions of rays [N_rays, 3].
        near: Near bound for sampling.
        far: Far bound for sampling.
        N_samples: Number of samples along each ray.
        rand: If True, perturb sample positions.
        lindisp: If True, sample linearly in disparity.
        use_viewdirs: If True, use view directions in NeRF model.
        white_bkgd: If True, composite RGB on a white background.
    Returns:
        A dictionary containing:
        'rgb_map': [N_rays, 3]. Estimated RGB color of a ray.
        'depth_map': [N_rays]. Estimated distance to object.
        'disp_map': [N_rays]. Estimated disparity.
        'acc_map': [N_rays]. Accumulated alpha.
        'weights': [N_rays, N_samples]. Weights assigned to each sample point.
        'z_vals': [N_rays, N_samples]. Depth of each sample point.
        'alpha': [N_rays, N_samples]. Alpha values.
        'raw_sigma': [N_rays, N_samples]. Raw sigma output from model.
    """
    N_rays = rays_o.shape[0]

    # Determine z_vals for sampling
    if z_vals_override is not None:
        z_vals = z_vals_override
        N_samples = z_vals.shape[-1] # Update N_samples to actual number of samples from override
    else:
        t_vals = torch.linspace(0., 1., steps=N_samples, device=rays_o.device) # Shape: [N_samples]

        # Handle scalar or per-ray near/far
        _near = near.view(-1, 1) if torch.is_tensor(near) and near.ndim > 0 and near.numel() > 1 else torch.tensor(near, device=rays_o.device).view(1,1)
        _far = far.view(-1, 1) if torch.is_tensor(far) and far.ndim > 0 and far.numel() > 1 else torch.tensor(far, device=rays_o.device).view(1,1)


        if lindisp:
            z_vals_flat = 1. / (1. / _near * (1. - t_vals.view(1, -1)) + 1. / _far * t_vals.view(1, -1))
        else:
            z_vals_flat = _near * (1. - t_vals.view(1, -1)) + _far * t_vals.view(1, -1)

        if z_vals_flat.shape[0] == 1 and rays_o.shape[0] > 1:
            z_vals = z_vals_flat.expand(rays_o.shape[0], N_samples)
        else: # Handles cases where near/far might be per-ray, or N_rays is 1
             z_vals = z_vals_flat.reshape(rays_o.shape[0], N_samples) if z_vals_flat.numel() == N_rays * N_samples else z_vals_flat

    if rand:
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        t_rand = torch.rand(z_vals.shape, device=rays_o.device)
        z_vals = lower + (upper - lower) * t_rand

    # Compute 3D query points: pts = o + t*d
    # rays_o: [N_rays, 3] -> [N_rays, 1, 3]
    # rays_d: [N_rays, 3] -> [N_rays, 1, 3]
    # z_vals: [N_rays, N_samples] -> [N_rays, N_samples, 1]
    # pts: [N_rays, N_samples, 3]
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    # Query NeRF model
    # pts.view(-1, 3) -> [N_rays * N_samples, 3]
    embedded_pts = embed_fn_pts(pts.reshape(-1, 3)) # Shape: [N_rays*N_samples, input_ch_pts]

    if use_viewdirs:
        # Normalize view directions
        viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True) # [N_rays, 3]
        # Expand viewdirs to match pts: [N_rays, N_samples, 3]
        viewdirs_expanded = viewdirs[:, None, :].expand_as(pts)
        # Embed view directions
        embedded_views = embed_fn_views(viewdirs_expanded.reshape(-1, 3)) # Shape: [N_rays*N_samples, input_ch_views]
        raw_output = nerf_model(embedded_pts, embedded_views)
    else:
        raw_output = nerf_model(embedded_pts)

    # Reshape raw_output to [N_rays, N_samples, 4] (3 for RGB, 1 for sigma)
    raw_output = raw_output.view(N_rays, N_samples, raw_output.shape[-1])

    # Extract RGB and sigma, apply activations
    rgb = torch.sigmoid(raw_output[..., :3])  # [N_rays, N_samples, 3]
    # sigma_a is density. ReLU ensures it's non-negative.
    sigma_a = F.relu(raw_output[..., 3])  # [N_rays, N_samples]
    raw_sigma_for_return = raw_output[...,3] # For inspection

    # Compute alpha values (volumetric rendering equation)
    # dists: [N_rays, N_samples]
    dists = z_vals[..., 1:] - z_vals[..., :-1] # Distances between adjacent samples
    # Last segment: distance from last sample to infinity (or a very large number)
    # This is to account for any density beyond the last sample.
    # A large distance means that if there's any density, it will fully occlude.
    dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10, device=rays_o.device)], -1)

    # alpha = 1 - exp(-sigma * delta)
    alpha = 1. - torch.exp(-sigma_a * dists)  # [N_rays, N_samples]

    # Compute weights (transmittance * alpha)
    # T_i = product_{j=1 to i-1} (1 - alpha_j)
    # weights_i = T_i * alpha_i
    # Transmittance T is the probability that the ray travels from near plane to sample i without being absorbed.
    # We use 1e-10 to avoid numerical instability (e.g. T becoming exactly 0 then log(T) is -inf).
    T = torch.cumprod(torch.cat([
        torch.ones((N_rays, 1), device=rays_o.device), # T_1 = 1 (ray has not hit anything yet)
        1. - alpha + 1e-10 # (1 - alpha_j) for j=1 to N_samples-1
    ], -1), -1)[:, :-1] # Exclude last element, T should have N_samples elements.
                        # Resulting T is [N_rays, N_samples]

    weights = alpha * T  # [N_rays, N_samples]

    # Compute accumulated maps
    # C(r) = sum_{i=1 to N} w_i * c_i
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # [N_rays, 3]

    # D(r) = sum_{i=1 to N} w_i * z_i
    depth_map = torch.sum(weights * z_vals, dim=-1) # [N_rays]

    # Accumulation map (total transmittance along the ray, or 1 - opacity)
    # acc_map = sum w_i. If all weights are small, acc_map is small (transparent).
    # If some weights are large, acc_map approaches 1 (opaque).
    acc_map = torch.sum(weights, dim=-1) # [N_rays]

    # Disparity map = 1 / depth_map (or more stable version)
    # Max is to prevent division by zero if depth_map is zero (e.g. if acc_map is zero)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / (acc_map + 1e-10))
    disp_map = torch.nan_to_num(disp_map, nan=0.0) # Handle cases where acc_map is 0, depth_map is 0 -> nan

    if white_bkgd:
        # Composite onto a white background
        # rgb_map = C_ray + (1 - acc_map) * white_color (which is 1.0)
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return {
        'rgb_map': rgb_map,
        'depth_map': depth_map,
        'disp_map': disp_map,
        'acc_map': acc_map,
        'weights': weights, # [N_rays, N_samples]
        'z_vals': z_vals,   # [N_rays, N_samples]
        'alpha': alpha,     # [N_rays, N_samples]
        'raw_sigma': raw_sigma_for_return # [N_rays, N_samples] for debugging/loss
    }
