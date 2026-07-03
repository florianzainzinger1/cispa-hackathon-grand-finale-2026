"""StyleGAN2 (CelebA-HQ) generative prior for the r1 risk branch.

Loads a pretrained NVlabs `stylegan2-ada-pytorch` generator in *no-plugin*
(pure-PyTorch) mode. This works on the JURECA stack (torch 2.12+cu130, no
ninja / nvcc / torchvision) because:
  * `bias_act` / `upfirdn2d` wrap their CUDA-plugin build in try/except and fall
    back to a pure-PyTorch `_ref` implementation -- we force that path by stubbing
    their `_init` to return False, so no build is ever attempted;
  * `conv2d_gradfix` is disabled-by-default and version-gated, so synthesis uses
    native `F.conv2d`, which supports create_graph double-backward -- exactly what
    latent gradient-matching needs.

We keep G frozen and optimize LATENT codes (W or W+). Constraining the search to
the face manifold is the ingredient that pixel-space batch-128 matching lacked
(v3's cosine-matched reconstructions collapsed to noise).

Env overrides:
  FL_SG_REPO : path to the stylegan2-ada-pytorch checkout
  FL_SG_PKL  : path to the generator .pkl (expects a dict with key 'G_ema')
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = "/p/scratch/training2625/zainzinger1/1337/recon_work/runs"
DEFAULT_REPO = os.path.join(_ROOT, "repos", "stylegan2-ada-pytorch")
DEFAULT_PKL = os.path.join(_ROOT, "weights", "celebahq256.pkl")


def load_generator(pkl=None, repo=None, device="cpu"):
    """Load and freeze an NVlabs G_ema generator in no-plugin mode."""
    repo = repo or os.environ.get("FL_SG_REPO", DEFAULT_REPO)
    pkl = pkl or os.environ.get("FL_SG_PKL", DEFAULT_PKL)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    # Force pure-PyTorch op path: never attempt to JIT-build CUDA kernels.
    import torch_utils.ops.bias_act as _ba
    _ba._init = lambda: False
    import torch_utils.ops.upfirdn2d as _up
    _up._init = lambda: False
    import legacy  # NVlabs pickle loader (uses pickle, not torch.load weights_only)
    with open(pkl, "rb") as f:
        G = legacy.load_network_pkl(f)["G_ema"]
    G = G.to(device).eval()
    for p in G.parameters():
        p.requires_grad_(False)
    return G


class StyleGANPrior(nn.Module):
    """Differentiable [0,1] face synthesizer at a target resolution.

    synth(w) -> (n, 3, out_size, out_size) in [0,1], differentiable w.r.t. w
    (double-backward safe). W+ latents have shape (n, num_ws, w_dim); W latents
    (n, w_dim) are broadcast across layers.
    """

    def __init__(self, G, out_size=64, w_plus=True):
        super().__init__()
        self.G = G
        self.out_size = int(out_size)
        self.w_plus = bool(w_plus)
        self.num_ws = int(G.num_ws)
        self.w_dim = int(G.w_dim)
        self.z_dim = int(G.z_dim)
        self.native_res = int(G.img_resolution)
        self.register_buffer("w_avg", G.mapping.w_avg.detach().clone())

    # -- latent initialisation ------------------------------------------------
    def init_w(self, n, device, jitter=0.05, from_mean=True):
        """Return a fresh latent (requires_grad=True). Seed torch RNG outside for
        reproducibility across restarts."""
        if from_mean:
            base = self.w_avg.to(device)
            if self.w_plus:
                w = base[None, None, :].repeat(n, self.num_ws, 1).clone()
            else:
                w = base[None, :].repeat(n, 1).clone()
            if jitter > 0:
                w = w + jitter * torch.randn(w.shape, device=device)
        else:
            z = torch.randn(n, self.z_dim, device=device)
            w = self.G.mapping(z, None)              # (n, num_ws, w_dim)
            if not self.w_plus:
                w = w[:, 0, :].contiguous()
        return w.detach().requires_grad_(True)

    @torch.no_grad()
    def sample(self, n, device, truncation=0.7):
        """Sample n random faces (no grad) for diversity fill / previews."""
        z = torch.randn(n, self.z_dim, device=device)
        w = self.G.mapping(z, None, truncation_psi=truncation)
        return self._synth_ws(w)

    @torch.no_grad()
    def w_radius(self, device, n=1024, pct=0.9):
        """Percentile of ||w - w_avg|| over sampled W codes -- calibrates a
        truncation ball that keeps optimised latents in the natural face region."""
        z = torch.randn(n, self.z_dim, device=device)
        w = self.G.mapping(z, None)[:, 0, :]
        d = (w - self.w_avg.to(device)).norm(dim=1)
        return float(torch.quantile(d, pct).item())

    # -- synthesis ------------------------------------------------------------
    def _synth_ws(self, w_ws):
        img = self.G.synthesis(w_ws, noise_mode="const", force_fp32=True)
        img = (img + 1.0) * 0.5
        img = img.clamp(0, 1)
        if img.shape[-1] != self.out_size:
            img = F.interpolate(img, size=(self.out_size, self.out_size), mode="area")
        return img

    def synth(self, w):
        if self.w_plus:
            w_ws = w
        else:
            w_ws = w[:, None, :].repeat(1, self.num_ws, 1)
        return self._synth_ws(w_ws)

    def w_prior_loss(self, w):
        """Mean squared distance of the latent from w_avg (keeps faces plausible)."""
        wa = self.w_avg.to(w.device)
        if self.w_plus:
            return ((w - wa[None, None, :]) ** 2).mean()
        return ((w - wa[None, :]) ** 2).mean()
