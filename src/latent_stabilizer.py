import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentStabilizer(nn.Module):
    """
    Advanced Stabilization Module for DreamDojo 2B:
    1. Scheme 2: Physical Ground-Truth Blending (Full-field continuous or radial Gaussian)
    2. Scheme 3: Latent Manifold Projection & Variance Clamping
    3. Mathematical 2D Binomial Anti-Checkerboard Low-Pass Filter:
       Eliminates 2x2 patch high-frequency Nyquist resonance ((-1)^x * (-1)^y)
    """
    def __init__(
        self,
        vae_config_path="/mnt/d/DreamDojo_2B/vae/config.json",
        device="cuda",
        dtype=torch.bfloat16,
        h_dim=30,
        w_dim=30,
        mask_mode="continuous", # 'continuous', 'radial_gaussian', or 'feathered_box'
        box_bounds=(0.20, 0.80, 0.20, 0.80),
        feather_margin=0.15,
        gaussian_center=(0.5, 0.5),
        gaussian_sigma=0.35,
        proj_strength=0.35,
        clamp_k=3.5,
        enable_anti_checkerboard=True,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype
        self.h_dim = h_dim
        self.w_dim = w_dim
        self.mask_mode = mask_mode
        self.proj_strength = proj_strength
        self.clamp_k = clamp_k
        self.enable_anti_checkerboard = enable_anti_checkerboard

        # 1. Load VAE statistical priors
        if not os.path.exists(vae_config_path):
            alt_path = "D:\\DreamDojo_2B\\vae\\config.json"
            if os.path.exists(alt_path):
                vae_config_path = alt_path
            else:
                raise FileNotFoundError(f"Cannot find VAE config at {vae_config_path} or {alt_path}")

        with open(vae_config_path, "r") as f:
            cfg = json.load(f)

        mean_list = cfg.get("latents_mean", [0.0] * 16)
        std_list = cfg.get("latents_std", [1.0] * 16)

        target_mean = torch.tensor(mean_list, dtype=self.dtype, device=self.device).view(1, 16, 1, 1, 1)
        target_std = torch.tensor(std_list, dtype=self.dtype, device=self.device).view(1, 16, 1, 1, 1)
        
        self.register_buffer("target_mean", target_mean)
        self.register_buffer("target_std", target_std)

        # Clamping bounds: [1, 16, 1, 1, 1]
        self.register_buffer("lower_bound", target_mean - clamp_k * target_std)
        self.register_buffer("upper_bound", target_mean + clamp_k * target_std)

        # 2. Build 2D Binomial Anti-Checkerboard Kernel (3x3 depthwise separable)
        # K = [1 2 1]^T * [1 2 1] / 16. Perfectly annihilates Nyquist checkerboard (-1)^x * (-1)^y
        k1d = torch.tensor([0.25, 0.5, 0.25], dtype=self.dtype, device=self.device)
        k2d = (k1d[:, None] * k1d[None, :]).view(1, 1, 3, 3)
        binomial_kernel = k2d.repeat(16, 1, 1, 1) # 16 groups
        self.register_buffer("binomial_kernel", binomial_kernel)

        # 3. Build Dynamic Mask (Scheme 2)
        if mask_mode == "continuous":
            # Seamless full-field continuous blending: no square box!
            # Uniform smooth interaction field
            dynamic_mask = torch.full((1, 1, 1, h_dim, w_dim), 0.45, dtype=self.dtype, device=self.device)
        elif mask_mode == "radial_gaussian":
            # Smooth circular Gaussian: zero square edges or corners
            y = torch.linspace(0.0, 1.0, h_dim, device=self.device)
            x = torch.linspace(0.0, 1.0, w_dim, device=self.device)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            cy, cx = gaussian_center
            dist_sq = (yy - cy) ** 2 + (xx - cx) ** 2
            g_mask = torch.exp(-dist_sq / (2.0 * (gaussian_sigma ** 2)))
            dynamic_mask = g_mask.view(1, 1, 1, h_dim, w_dim).to(dtype=self.dtype)
        else: # feathered_box
            ymin, ymax, xmin, xmax = box_bounds
            mask_2d = self._build_smooth_box_mask(h_dim, w_dim, ymin, ymax, xmin, xmax, feather_margin)
            dynamic_mask = mask_2d.view(1, 1, 1, h_dim, w_dim).to(device=self.device, dtype=self.dtype)

        self.register_buffer("dynamic_mask", dynamic_mask)

    def _build_smooth_box_mask(self, H, W, ymin, ymax, xmin, xmax, margin):
        y = torch.linspace(0.0, 1.0, H, device=self.device)
        x = torch.linspace(0.0, 1.0, W, device=self.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        mask_y = torch.ones_like(yy)
        mask_y = torch.where(yy < ymin, 0.0, mask_y)
        trans_y_low = (yy >= ymin) & (yy < ymin + margin)
        mask_y[trans_y_low] = 0.5 * (1.0 - torch.cos(torch.pi * (yy[trans_y_low] - ymin) / margin))
        trans_y_high = (yy > ymax - margin) & (yy <= ymax)
        mask_y[trans_y_high] = 0.5 * (1.0 + torch.cos(torch.pi * (yy[trans_y_high] - (ymax - margin)) / margin))
        mask_y = torch.where(yy > ymax, 0.0, mask_y)

        mask_x = torch.ones_like(xx)
        mask_x = torch.where(xx < xmin, 0.0, mask_x)
        trans_x_low = (xx >= xmin) & (xx < xmin + margin)
        mask_x[trans_x_low] = 0.5 * (1.0 - torch.cos(torch.pi * (xx[trans_x_low] - xmin) / margin))
        trans_x_high = (xx > xmax - margin) & (xx <= xmax)
        mask_x[trans_x_high] = 0.5 * (1.0 + torch.cos(torch.pi * (xx[trans_x_high] - (xmax - margin)) / margin))
        mask_x = torch.where(xx > xmax, 0.0, mask_x)

        return (mask_y * mask_x).clamp(0.0, 1.0)

    def apply_anti_checkerboard(self, z):
        """
        Applies depthwise binomial low-pass filtering across all 16 latent channels.
        Mathematically cancels out the 2x2 patch alternating checkerboard mode.
        """
        B, C, T, H, W = z.shape
        # Permute to [B*T, C, H, W] for 2D convolution
        z_2d = z.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        
        # Two-pass binomial blur with replicate padding completely erases checkerboard
        pad = F.pad(z_2d, (1, 1, 1, 1), mode="replicate")
        z_smooth = F.conv2d(pad, self.binomial_kernel, groups=C)
        pad2 = F.pad(z_smooth, (1, 1, 1, 1), mode="replicate")
        z_smooth2 = F.conv2d(pad2, self.binomial_kernel, groups=C)

        return z_smooth2.view(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()

    def apply_static_freezing(self, z_pred, z_canonical):
        """
        Scheme 2: Blend dynamic prediction with real D455 canonical reference frame.
        """
        return self.dynamic_mask * z_pred + (1.0 - self.dynamic_mask) * z_canonical

    def apply_manifold_projection(self, z):
        """
        Scheme 3: Re-anchor channel-wise spatial statistics to Wan-VAE training distribution
        and clamp unphysical outliers.
        """
        curr_mean = z.mean(dim=(-2, -1), keepdim=True)
        curr_std = z.std(dim=(-2, -1), keepdim=True) + 1e-6

        z_projected = (z - curr_mean) / curr_std * self.target_std + self.target_mean

        if self.proj_strength < 1.0:
            z_stabilized = (1.0 - self.proj_strength) * z + self.proj_strength * z_projected
        else:
            z_stabilized = z_projected

        z_stabilized = torch.max(torch.min(z_stabilized, self.upper_bound), self.lower_bound)
        return z_stabilized

    def forward(self, z_pred, z_canonical, enable_scheme2=True, enable_scheme3=True):
        out = z_pred
        
        # 1. Apply anti-checkerboard spatial filter to extinguish patch resonance
        if self.enable_anti_checkerboard:
            out = self.apply_anti_checkerboard(out)
            
        # 2. Scheme 2: Physical Ground Truth Blending
        if enable_scheme2 and z_canonical is not None:
            out = self.apply_static_freezing(out, z_canonical)
            
        # 3. Scheme 3: Manifold Projection & Statistics Clamping
        if enable_scheme3:
            out = self.apply_manifold_projection(out)
            
        return out
