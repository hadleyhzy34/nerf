"""Core NeRF model components, including Positional Encoding, the NeRF MLP,
volume rendering functions, and hierarchical sampling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PositionalEncoder(nn.Module):
    """Implements positional encoding as described in the NeRF paper.

    This module takes an input tensor and maps it to a higher-dimensional
    space using a set of sinusoidal functions of varying frequencies.
    This allows the MLP to learn higher-frequency variations in the data.

    Attributes:
        input_dims (int): Dimensionality of the input tensor (e.g., 3 for coordinates).
        num_freqs (int): Number of frequency bands (L in the paper).
        log_sampling (bool): If True, frequencies are sampled logarithmically.
                             If False, frequencies are sampled linearly.
        output_dims (int): Dimensionality of the encoded output.
        freq_bands (torch.Tensor): Tensor of frequency bands.
    """
    def __init__(self, input_dims: int, num_freqs: int, log_sampling: bool = True):
        """Initializes the PositionalEncoder.

        Args:
            input_dims (int): Dimensionality of the input (e.g., 3 for coordinates,
                              2 or 3 for viewing directions).
            num_freqs (int): Number of frequencies L for the encoding
                             (sin(2^l * pi * p), cos(2^l * pi * p)).
            log_sampling (bool, optional): Whether to sample frequencies in log scale
                                           or linear scale. Defaults to True.
        """
        super().__init__()
        self.input_dims = input_dims
        self.num_freqs = num_freqs
        self.log_sampling = log_sampling

        # Create frequency bands
        if self.log_sampling:
            # Frequencies are 2^0, 2^1, ..., 2^(L-1)
            freq_bands_init = 2.**torch.linspace(0., num_freqs - 1, num_freqs)
        else:
            # Frequencies are linearly spaced between 1 and 2^(L-1)
            freq_bands_init = torch.linspace(1., 2.**(num_freqs - 1), num_freqs)

        # Register freq_bands as a buffer so it's part of the module's state
        # and moves with model.to(device)
        self.register_buffer('freq_bands', freq_bands_init)

        # Calculate output dimensions: original input + (sin, cos for each freq)
        self.output_dims = self.input_dims * (1 + 2 * self.num_freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies positional encoding to the input tensor.

        The input `x` is expected to have shape [N, ..., input_dims], where N is
        the batch size and "..." represents any number of intermediate dimensions.
        The encoding is applied to the last dimension.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Positionally encoded tensor of shape [N, ..., output_dims].
        """
        outputs = [x] # Start with the original input
        # Apply sin and cos functions for each frequency band
        for freq in self.freq_bands:
            # x * freq will broadcast correctly if x is [..., input_dims]
            outputs.append(torch.sin(x * freq))
            outputs.append(torch.cos(x * freq))

        # Concatenate all encoded features along the last dimension
        return torch.cat(outputs, dim=-1)


class NeRF(nn.Module):
    """Neural Radiance Field (NeRF) model.

    This model takes encoded 3D coordinates (and optionally encoded viewing directions)
    and outputs the color (RGB) and volume density (sigma) at those points.
    The architecture follows the original NeRF paper.

    Attributes:
        D (int): Number of linear layers in the main network processing spatial coordinates.
        W (int): Number of hidden units (width) per layer.
        input_ch_pts (int): Number of channels for the encoded input 3D coordinates.
        input_ch_views (int): Number of channels for the encoded input viewing directions.
        skips (list[int]): List of layer indices where residual/skip connections should be added.
                           The skip connection concatenates the original encoded points.
        use_viewdirs (bool): If True, the model uses viewing directions to predict RGB color.
        pts_linears (nn.ModuleList): MLP for processing spatial coordinates.
        sigma_linear (nn.Linear, optional): Final layer to predict volume density (sigma) if using viewdirs.
        feature_linear (nn.Linear, optional): Layer to produce features for RGB prediction if using viewdirs.
        rgb_layers (nn.Sequential, optional): MLP for predicting RGB from features and view directions.
        output_linear (nn.Linear, optional): Final layer for RGB and sigma if not using viewdirs.
    """
    def __init__(self, D: int = 8, W: int = 256,
                 input_ch_pts: int = 63, input_ch_views: int = 27,
                 skips: list[int] = [4], use_viewdirs: bool = True):
        """Initializes the NeRF model.

        Args:
            D (int, optional): Number of linear layers in the main network (sigma prediction path). Defaults to 8.
            W (int, optional): Number of hidden units per layer. Defaults to 256.
            input_ch_pts (int, optional): Number of input channels for encoded 3D coordinates.
                                         (e.g., 3 + 3*2*L_pts for positional encoding). Defaults to 63.
            input_ch_views (int, optional): Number of input channels for encoded viewing directions.
                                           (e.g., 3 + 3*2*L_views for positional encoding). Defaults to 27.
                                           Set to 0 if `use_viewdirs` is False.
            skips (list[int], optional): List of layer indices (0-indexed) after which the original
                                         encoded points input should be concatenated. Defaults to [4].
            use_viewdirs (bool, optional): Whether to use viewing directions for color prediction. Defaults to True.
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch_pts = input_ch_pts
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        # MLP for processing spatial coordinates (sigma and feature path)
        self.pts_linears = nn.ModuleList()
        # First layer
        self.pts_linears.append(nn.Linear(input_ch_pts, W))
        # Subsequent D-1 layers
        for i in range(D - 1):
            if i in self.skips:
                # Layer takes current W channels + original input_ch_pts from skip connection
                self.pts_linears.append(nn.Linear(W + input_ch_pts, W))
            else:
                self.pts_linears.append(nn.Linear(W, W))

        # Layers for outputting RGB and sigma
        if use_viewdirs:
            # Sigma is predicted from the main path's features
            self.sigma_linear = nn.Linear(W, 1)
            # Feature vector for RGB prediction also comes from the main path
            self.feature_linear = nn.Linear(W, W)

            # MLP for predicting RGB from features and view directions
            # Takes concatenated features (W) and encoded view directions (input_ch_views)
            self.rgb_layers = nn.Sequential(
                nn.Linear(W + input_ch_views, W // 2),
                nn.ReLU(),
                nn.Linear(W // 2, 3) # Output 3 channels for RGB
            )
        else:
            # If not using view directions, predict RGB and sigma directly from the main network's features
            self.output_linear = nn.Linear(W, 4)  # Output: RGB (3) + Sigma (1)

    def forward(self, x_pts_encoded: torch.Tensor, x_views_encoded: torch.Tensor = None) -> torch.Tensor:
        """Forward pass of the NeRF model.

        Args:
            x_pts_encoded (torch.Tensor): Positionally encoded 3D coordinates.
                                          Shape: [batch_size, ..., input_ch_pts].
            x_views_encoded (torch.Tensor, optional): Positionally encoded viewing directions.
                                                      Shape: [batch_size, ..., input_ch_views].
                                                      Required if `self.use_viewdirs` is True.
                                                      Defaults to None.

        Returns:
            torch.Tensor: Output tensor containing RGB color (3 channels) and volume
                          density sigma (1 channel). Shape: [batch_size, ..., 4].

        Raises:
            ValueError: If `use_viewdirs` is True but `x_views_encoded` is not provided.
        """
        h = x_pts_encoded # 'h' holds the current features

        # Pass through the main MLP (pts_linears)
        for i, layer in enumerate(self.pts_linears):
            h = layer(h)
            h = F.relu(h)
            # Apply skip connection: concatenate original encoded points
            if i in self.skips:
                # Note: The skip connection is applied *after* the ReLU of layer `i`.
                # The concatenated result is then fed into layer `i+1`.
                # This means layer `i+1` must be defined to accept `W + input_ch_pts`.
                h = torch.cat([x_pts_encoded, h], dim=-1)

        # Predict sigma and RGB based on whether view directions are used
        if self.use_viewdirs:
            if x_views_encoded is None:
                raise ValueError("View directions (x_views_encoded) must be provided when use_viewdirs is True.")

            # Sigma is predicted from the final features of the main MLP
            sigma = self.sigma_linear(h)

            # Features for RGB prediction are also derived from these features
            features = self.feature_linear(h)

            # Concatenate features with encoded view directions
            combined = torch.cat([features, x_views_encoded], dim=-1)

            # Predict RGB using the RGB-specific layers
            rgb = self.rgb_layers(combined)
        else:
            # Predict RGB and sigma directly from the final features
            outputs = self.output_linear(h)
            rgb = outputs[..., :3]
            sigma = outputs[..., 3:4] # Ensure sigma maintains shape [..., 1]

        # Concatenate RGB and sigma to produce the final output [..., 4]
        return torch.cat([rgb, sigma], dim=-1)


# Hierarchical sampling helper: Inverse transform sampling
def sample_pdf(bins: torch.Tensor, weights: torch.Tensor, N_samples: int, det: bool = False) -> torch.Tensor:
    """Performs inverse transform sampling using a piece-wise constant PDF.

    This function is used in hierarchical volume sampling to sample more points
    in regions where the NeRF model predicts high density (large weights).

    Args:
        bins (torch.Tensor): Defines the edges of the bins along each ray.
                             Shape: [N_rays, N_bins_coarse - 1]. These are typically
                             the midpoints of the `z_vals` from the coarse pass.
        weights (torch.Tensor): Weights assigned to each bin, used to construct the PDF.
                                Shape: [N_rays, N_bins_coarse - 1]. Typically these are
                                `weights_coarse[..., 1:-1]` from `render_rays`.
        N_samples (int): Number of new samples to draw per ray (N_importance).
        det (bool, optional): If True, performs deterministic sampling by taking
                              linearly spaced samples in the CDF. If False, performs
                              stochastic sampling by drawing random uniform samples.
                              Defaults to False.

    Returns:
        torch.Tensor: New samples `z_vals` drawn from the PDF. Shape: [N_rays, N_samples].
    """
    # Add a small epsilon to weights to prevent division by zero and ensure all bins have some probability.
    # This also handles cases where all weights might be zero for a ray.
    weights = weights + 1e-5  # Shape: [N_rays, N_bins_coarse - 1]

    # Normalize weights to get the PDF
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)  # Shape: [N_rays, N_bins_coarse - 1]

    # Compute CDF from PDF
    cdf = torch.cumsum(pdf, dim=-1)  # Shape: [N_rays, N_bins_coarse - 1]

    # Prepend zeros to the CDF to define the start of the first bin at CDF=0.
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # Shape: [N_rays, N_bins_coarse] (now N_bins_coarse intervals)

    # Generate uniform samples `u` for inverse transform sampling.
    if det:
        # Deterministic sampling: uniformly spaced samples in [0, 1]
        u = torch.linspace(0., 1., steps=N_samples, device=bins.device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples]) # Shape: [N_rays, N_samples]
    else:
        # Stochastic sampling: random samples from Uniform(0, 1)
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=bins.device) # Shape: [N_rays, N_samples]

    u = u.contiguous() # Ensure tensor is contiguous for `torch.searchsorted`.

    # Find the indices where `u` would be inserted into `cdf` to maintain order.
    # `right=True` means that if u[i] == cdf[j], then inds[i] = j+1.
    # This gives the index of the CDF bin that each `u` falls into.
    inds = torch.searchsorted(cdf, u, right=True) # Shape: [N_rays, N_samples]

    # Calculate lower and upper bin indices for linear interpolation.
    # Clamp indices to be within the valid range of CDF array.
    below = torch.max(torch.zeros_like(inds - 1), inds - 1) # Shape: [N_rays, N_samples]
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds) # Shape: [N_rays, N_samples]

    # Stack `below` and `above` to gather corresponding CDF values and bin edges.
    inds_g = torch.stack([below, above], dim=-1)  # Shape: [N_rays, N_samples, 2]

    # Gather CDF values (cdf_g) and bin edges (bins_g) for interpolation.
    # `cdf` is [N_rays, N_bins_coarse], `inds_g` indices into the last dim of cdf.
    # `bins` is [N_rays, N_bins_coarse - 1].
    # Need to ensure `bins` aligns with `cdf` if `cdf` had a 0 prepended.
    # If `bins` are midpoints, they align with `pdf`.
    # The `cdf` has N_bins_coarse elements (0, cdf_1, cdf_2, ..., cdf_{N-1}).
    # The `bins` (from which we sample) should correspond to these CDF values.
    # If `bins` are the original z_vals midpoints [z_mid_0, ..., z_mid_{N-2}],
    # and cdf is [0, cdf_0, ..., cdf_{N-2}], then we need N-1 bins.
    # The `inds_g` indices are into the `cdf` array of size `N_bins_coarse`.
    # We need to gather from `cdf` and also from a version of `bins` that aligns with `cdf` intervals.
    # Let the input `bins` be the left edges of the intervals.
    # If cdf is [c0, c1, c2, c3] and bins are [b0, b1, b2], then (c0,c1) maps to b0, (c1,c2) to b1 etc.
    # A common way is to have `bins` be the z-values at the *edges* of the PDF intervals.
    # If `bins` are midpoints `z_vals_mid` (N-1 points), and `weights` (N-1 values) correspond to these intervals.
    # `cdf` becomes `[0, cdf_0, ..., cdf_{N-2}]` (N points).
    # `inds_g` indexes into this `cdf`.
    # `bins_g` should gather from the original `bins` (midpoints).
    # The `sample_pdf` in official NeRF uses `bins = z_vals_mid`.

    batch_size = cdf.shape[0]
    # `gather` expects index tensor to have same number of dims as input tensor for non-indexed dims.
    cdf_g = torch.gather(cdf, dim=1, index=inds_g.view(batch_size, -1)).view(batch_size, N_samples, 2)
    bins_g = torch.gather(bins, dim=1, index=inds_g.view(batch_size, -1)).view(batch_size, N_samples, 2)
    # This assumes `bins` has the same number of elements along dim 1 as `cdf` for gathering.
    # If `bins` are midpoints (N_samples_coarse - 1), and `cdf` is (N_samples_coarse),
    # then `inds_g` (derived from `cdf`) might point out of bounds for `bins` if not careful.
    # However, `torch.searchsorted` on `cdf` of size K gives indices from 0 to K.
    # `inds` are indices for `cdf`. `below` and `above` are also for `cdf`.
    # If `bins` are actual z-values at the edges of these CDF intervals, this is fine.
    # Original NeRF `sample_pdf` uses `z_vals[..., :-1]` and `z_vals[..., 1:]` as `bin_edges_low` and `bin_edges_high`
    # and then uses `inds` to pick from these.
    # Here, `bins` are midpoints. The logic `bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])`
    # interpolates between two *midpoints*. This is standard for piece-wise constant PDF.

    # Linearly interpolate within each selected bin to get the new sample value.
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    # Avoid division by zero if a CDF interval has zero probability mass (cdf_g[...,1] == cdf_g[...,0]).
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom) # Use 1 where denom is too small

    t = (u - cdf_g[..., 0]) / denom  # Interpolation weight, in [0, 1]
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])  # Linearly interpolate between bin midpoints

    return samples


def render_rays(
    nerf_model: NeRF,
    embed_fn_pts,
    embed_fn_views,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float, # Or torch.Tensor for per-ray near
    far: float,  # Or torch.Tensor for per-ray far
    N_samples: int,
    rand: bool = False,
    lindisp: bool = False,
    use_viewdirs: bool = True,
    white_bkgd: bool = False,
    z_vals_override: torch.Tensor = None
) -> dict[str, torch.Tensor]:
    """Renders rays using the NeRF model via volumetric rendering.

    Args:
        nerf_model (NeRF): The NeRF MLP model to query.
        embed_fn_pts (callable): Positional encoding function for 3D points (xyz).
        embed_fn_views (callable, optional): Positional encoding function for view directions.
                                             Required if `use_viewdirs` is True.
        rays_o (torch.Tensor): Origins of the rays. Shape: [N_rays, 3].
        rays_d (torch.Tensor): Directions of the rays (normalized). Shape: [N_rays, 3].
        near (float or torch.Tensor): Near sampling bound. Can be a scalar or per-ray tensor [N_rays, 1].
        far (float or torch.Tensor): Far sampling bound. Can be a scalar or per-ray tensor [N_rays, 1].
        N_samples (int): Number of samples to take along each ray. Used if `z_vals_override` is None.
        rand (bool, optional): If True, perturb sample positions along rays (stratified sampling).
                               Defaults to False.
        lindisp (bool, optional): If True, sample linearly in disparity rather than depth.
                                  Defaults to False.
        use_viewdirs (bool, optional): If True, the `nerf_model` uses view directions.
                                       Defaults to True.
        white_bkgd (bool, optional): If True, composite RGB color onto a white background
                                     if the ray does not hit anything (alpha accumulation < 1).
                                     Defaults to False (composites on black).
        z_vals_override (torch.Tensor, optional): If provided, these specific z-values are used for sampling
                                                  along rays, overriding `near`, `far`, `N_samples`, `lindisp`.
                                                  Shape: [N_rays, N_override_samples]. Defaults to None.

    Returns:
        dict[str, torch.Tensor]: A dictionary containing the rendering results:
            'rgb_map' (torch.Tensor): Estimated RGB color for each ray. [N_rays, 3].
            'depth_map' (torch.Tensor): Estimated depth for each ray. [N_rays].
            'disp_map' (torch.Tensor): Estimated disparity (1/depth) for each ray. [N_rays].
            'acc_map' (torch.Tensor): Accumulated alpha/opacity for each ray. [N_rays].
            'weights' (torch.Tensor): Weights assigned to each sample point along rays. [N_rays, N_samples_actual].
            'z_vals' (torch.Tensor): Depth of each sample point. [N_rays, N_samples_actual].
            'alpha' (torch.Tensor): Alpha values for each sample. [N_rays, N_samples_actual].
            'raw_sigma' (torch.Tensor): Raw sigma output from the model for each sample. [N_rays, N_samples_actual].
    """
    N_rays = rays_o.shape[0]
    N_samples_actual = N_samples # This might change if z_vals_override is used

    # Determine z_vals (depths for sampling) along each ray
    if z_vals_override is not None:
        z_vals = z_vals_override
        N_samples_actual = z_vals.shape[-1] # Update N_samples based on override
    else:
        t_vals = torch.linspace(0., 1., steps=N_samples, device=rays_o.device) # Shape: [N_samples]

        # Handle scalar or per-ray near/far bounds
        # Ensure _near and _far are [N_rays_or_1, 1] for broadcasting with t_vals [1, N_samples]
        _near = torch.as_tensor(near, device=rays_o.device).view(-1, 1)
        _far = torch.as_tensor(far, device=rays_o.device).view(-1, 1)
        if _near.shape[0] == 1 and N_rays > 1: _near = _near.expand(N_rays, 1)
        if _far.shape[0] == 1 and N_rays > 1: _far = _far.expand(N_rays, 1)


        if lindisp: # Sample linearly in disparity
            z_vals = 1. / (1. / _near * (1. - t_vals) + 1. / _far * t_vals)
        else: # Sample linearly in depth
            z_vals = _near * (1. - t_vals) + _far * t_vals
        # z_vals shape: [N_rays, N_samples]

    if rand: # Apply stratified sampling if specified
        # Get midpoints of z_vals intervals
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        # Define upper and lower bounds for sampling within each interval
        upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
        lower = torch.cat([z_vals[..., :1], mids], dim=-1)
        # Draw random samples from Uniform(0, 1)
        t_rand = torch.rand(z_vals.shape, device=rays_o.device)
        # Perturb z_vals within their intervals
        z_vals = lower + (upper - lower) * t_rand

    # Compute 3D query points: pts = o + t*d
    # rays_o: [N_rays, 3] -> [N_rays, 1, 3]
    # rays_d: [N_rays, 3] -> [N_rays, 1, 3]
    # z_vals: [N_rays, N_samples_actual] -> [N_rays, N_samples_actual, 1]
    # pts: [N_rays, N_samples_actual, 3]
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    # Query NeRF model for raw RGB and sigma values
    # Reshape points for batch processing by the MLP: [N_rays * N_samples_actual, 3]
    embedded_pts = embed_fn_pts(pts.reshape(-1, 3)) # Shape: [N_rays*N_samples_actual, input_ch_pts]

    if use_viewdirs:
        # Prepare view directions for the model
        viewdirs_norm = rays_d / torch.norm(rays_d, dim=-1, keepdim=True) # [N_rays, 3]
        # Expand viewdirs to match number of samples: [N_rays, N_samples_actual, 3]
        viewdirs_expanded = viewdirs_norm[:, None, :].expand_as(pts)
        # Embed view directions
        embedded_views = embed_fn_views(viewdirs_expanded.reshape(-1, 3)) # Shape: [N_rays*N_samples_actual, input_ch_views]
        raw_output = nerf_model(embedded_pts, embedded_views)
    else:
        raw_output = nerf_model(embedded_pts)

    # Reshape raw_output back to [N_rays, N_samples_actual, 4] (3 for RGB, 1 for sigma)
    raw_output = raw_output.view(N_rays, N_samples_actual, raw_output.shape[-1])

    # Apply activation functions to get RGB and sigma
    rgb = torch.sigmoid(raw_output[..., :3])  # RGB values in [0, 1]. Shape: [N_rays, N_samples_actual, 3]
    sigma_a = F.relu(raw_output[..., 3])      # Volume density (sigma). Shape: [N_rays, N_samples_actual]
    raw_sigma_for_return = raw_output[...,3]  # Raw sigma for potential inspection or other losses

    # Compute alpha values (discrete opacity) for volumetric rendering
    # dists: distances between adjacent samples along each ray. [N_rays, N_samples_actual]
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    # The last segment extends to "infinity" (represented by a large value).
    # This ensures that any density beyond the last sample contributes to occlusion.
    dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10, device=rays_o.device)], dim=-1)

    # alpha_i = 1 - exp(-sigma_i * delta_i), where delta_i is the distance (dists)
    alpha = 1. - torch.exp(-sigma_a * dists)  # Shape: [N_rays, N_samples_actual]

    # Compute weights for each sample: w_i = T_i * alpha_i
    # T_i = product_{j=1 to i-1} (1 - alpha_j) is the transmittance.
    # Add 1e-10 for numerical stability to prevent T from becoming exactly zero.
    transmittance = torch.cumprod(torch.cat([
        torch.ones((N_rays, 1), device=rays_o.device), # T_1 = 1 (transmittance before first sample)
        1. - alpha + 1e-10  # (1 - alpha_j) for j=1 to N_samples_actual-1
    ], dim=-1), dim=-1)[:, :-1] # Exclude last element. Shape: [N_rays, N_samples_actual]

    weights = alpha * transmittance  # Shape: [N_rays, N_samples_actual]

    # Compute final outputs by weighted summation (numerical integration)
    # rgb_map: Estimated RGB color for each ray. [N_rays, 3]
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)

    # depth_map: Estimated depth for each ray. [N_rays]
    depth_map = torch.sum(weights * z_vals, dim=-1)

    # acc_map: Accumulated alpha (total opacity) for each ray. [N_rays]
    # This is sum of weights, indicating how much light is absorbed/emitted.
    acc_map = torch.sum(weights, dim=-1)

    # disp_map: Estimated disparity (1/depth) for each ray. [N_rays]
    # Stabilized division for disparity.
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / (acc_map + 1e-10))
    disp_map = torch.nan_to_num(disp_map, nan=0.0) # Handle cases where acc_map is 0.

    if white_bkgd:
        # Composite RGB color onto a white background if specified.
        rgb_map = rgb_map + (1. - acc_map[..., None]) # Add (1-T_final)*background_color

    return {
        'rgb_map': rgb_map,
        'depth_map': depth_map,
        'disp_map': disp_map,
        'acc_map': acc_map,
        'weights': weights,
        'z_vals': z_vals,
        'alpha': alpha,
        'raw_sigma': raw_sigma_for_return
    }

# Example usage block (remains for testing, not part of the core API)
if __name__ == '__main__':
    # Example Usage for PositionalEncoder
    input_dim_pe = 3
    L_pe = 10
    encoder_pe = PositionalEncoder(input_dim_pe, L_pe)
    print(f"PositionalEncoder Output dimensions: {encoder_pe.output_dims}")
    test_input_pe = torch.rand(2, 5, input_dim_pe)
    encoded_output_pe = encoder_pe(test_input_pe)
    print(f"PE Input shape: {test_input_pe.shape}, PE Encoded output shape: {encoded_output_pe.shape}")

    # Example Usage for NeRF model
    L_pts_nerf = 10
    L_views_nerf = 4
    pts_encoder_nerf = PositionalEncoder(3, L_pts_nerf)
    views_encoder_nerf = PositionalEncoder(3, L_views_nerf)
    input_ch_pts_nerf = pts_encoder_nerf.output_dims
    input_ch_views_nerf = views_encoder_nerf.output_dims

    nerf_model_vd_test = NeRF(input_ch_pts=input_ch_pts_nerf, input_ch_views=input_ch_views_nerf, use_viewdirs=True)
    batch_size_nerf = 4
    num_points_nerf = 1024
    raw_pts_nerf = torch.rand(batch_size_nerf, num_points_nerf, 3)
    raw_views_nerf = torch.rand(batch_size_nerf, num_points_nerf, 3)
    raw_views_nerf = raw_views_nerf / torch.norm(raw_views_nerf, dim=-1, keepdim=True)
    encoded_pts_nerf = pts_encoder_nerf(raw_pts_nerf)
    encoded_views_nerf = views_encoder_nerf(raw_views_nerf)
    print(f"\n--- NeRF with view directions ---")
    output_vd_nerf = nerf_model_vd_test(encoded_pts_nerf, encoded_views_nerf)
    print(f"NeRF Output shape (with viewdirs): {output_vd_nerf.shape}")
    assert output_vd_nerf.shape == (batch_size_nerf, num_points_nerf, 4)

    nerf_model_no_vd_test = NeRF(input_ch_pts=input_ch_pts_nerf, use_viewdirs=False)
    print(f"\n--- NeRF without view directions ---")
    output_no_vd_nerf = nerf_model_no_vd_test(encoded_pts_nerf)
    print(f"NeRF Output shape (without viewdirs): {output_no_vd_nerf.shape}")
    assert output_no_vd_nerf.shape == (batch_size_nerf, num_points_nerf, 4)
    print("NeRF model tests seem okay.")
