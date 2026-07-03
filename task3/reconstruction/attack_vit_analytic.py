"""a1: ViT patch-embedding analytic inversion for models 9 and 11.

KEY INSIGHT:
The ViT patch_embed is Conv2d(3, dim, patch, stride=patch) — a non-overlapping
linear projection from each patch-sized RGB region to the embedding space. Its
gradient gW has shape (dim, 3*patch*patch). This is structurally identical to
the MLP first-layer gradient: gW[row] / gb[row] yields a weighted average of
the patch-pixel inputs.

DIFFERENCE FROM MLP:
The patch embedding is shared across ALL spatial positions. The gradient
aggregates over 128 images × (64/patch)² positions per image:
  model9:  128 images × 64 patches  = 8192 "samples", dim=192, patch_pixels=192
  model11: 128 images × 16 patches  = 2048 "samples", dim=256, patch_pixels=768

So for model9, gW is (192, 192) — exactly square, full rank possible, but the
128 images × 64 positions are heavily mixed in each gradient row.

STRATEGY:
1. SVD of patch-embedding gradient → top singular vectors = patch-shaped fragments
2. Ratio gW/gb per row → each ratio-row is a mixture of patch-pixel inputs
3. Cluster ratio-rows and extract representative patches
4. Use positional embedding gradient to assign patches to positions
5. Assemble patch jigsaw into full images
6. Quality filter and select 128 diverse reconstructions

ALTERNATIVE (simpler, high-value):
Even without full jigsaw reconstruction, we can use the patch-embedding SVD to
build a MUCH better warm-start for gradient matching than the face bank:
- Extract top singular vectors of gW as patch-shaped basis functions
- Random (but structured) combinations of these basis patches create plausible
  face-like starting points that are in the image subspace
- Warm-start the v3 gradient matching from these instead of random faces

This script implements both approaches.

Writes to candidates_a1/<model>/. Never touches existing candidates.
Login-node safe for the analytic parts.
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F

import utils as U

try:
    import vit as VIT
except Exception:
    VIT = None

EPS = 1e-8


# ---------------------------------------------------------------------------
# Patch embedding gradient analysis
# ---------------------------------------------------------------------------
def analyze_patch_embed(g, name, out_dir):
    """Analyze the patch embedding gradient and extract patch-space signal."""
    pw = g.get("patch_embed.proj.weight")
    pb = g.get("patch_embed.proj.bias")

    if pw is None:
        print(f"  {name}: no patch_embed.proj.weight gradient")
        return None

    dim = pw.shape[0]          # embedding dimension
    in_c = pw.shape[1]         # 3 (RGB)
    patch_h = pw.shape[2]      # patch size
    patch_w = pw.shape[3]

    patch_pixels = in_c * patch_h * patch_w  # 3 * patch * patch

    print(f"  Patch embed: dim={dim}, patch={patch_h}x{patch_w}, "
          f"pixels_per_patch={patch_pixels}")

    # Reshape to 2D gradient matrix: (dim, 3*patch*patch)
    Gp = pw.float().reshape(dim, patch_pixels)

    # SVD of the patch-embedding gradient
    U_, S, Vh = torch.linalg.svd(Gp, full_matrices=False)
    S = S.cpu()
    s0 = S[0].item()
    eff_rank = int((S > 0.01 * s0).sum().item())

    print(f"  Patch embed SVD: eff_rank={eff_rank}, "
          f"s[0]={s0:.4f}, s[{min(dim-1, 127)}]={S[min(dim-1, 127)].item():.4f}")

    # Save singular vectors as patch-shaped images
    k_show = min(64, dim)
    vecs = Vh[:k_show]  # (k, patch_pixels)
    patch_imgs = vecs.reshape(-1, in_c, patch_h, patch_w)
    patch_imgs = U.per_image_minmax(patch_imgs)
    # Upscale patches for visibility
    patch_imgs_up = F.interpolate(patch_imgs, size=(64, 64), mode="nearest")
    U.save_image_grid(patch_imgs_up, os.path.join(out_dir, "patch_singvecs.png"),
                      ncol=8)

    # Ratio analysis: gW[row] / gb[row] per patch-embedding neuron
    ratio_patches = None
    reliable = None
    if pb is not None:
        pb_f = pb.float()
        reliable = pb_f.abs() > EPS
        denom = torch.where(pb_f.abs() > EPS, pb_f,
                            torch.full_like(pb_f, float("nan")))
        ratio = Gp / denom.unsqueeze(1)  # (dim, patch_pixels)
        ratio_patches = ratio.reshape(dim, in_c, patch_h, patch_w)

        # Stats on ratio rows
        ratio_safe = torch.nan_to_num(ratio)
        rng_frac = ((ratio_safe >= -0.05) & (ratio_safe <= 1.05)).float().mean(dim=1)
        n_good = int((reliable & torch.isfinite(ratio).all(dim=1) &
                      (rng_frac > 0.5)).sum())
        print(f"  Ratio patches: {n_good}/{dim} good (in-range, reliable, finite)")

    return {
        "dim": dim, "patch_h": patch_h, "patch_w": patch_w,
        "patch_pixels": patch_pixels, "eff_rank": eff_rank,
        "S": S, "U_": U_.cpu(), "Vh": Vh.cpu(),
        "ratio_patches": ratio_patches,
        "reliable": reliable,
        "Gp": Gp.cpu(),
    }


# ---------------------------------------------------------------------------
# Position embedding analysis
# ---------------------------------------------------------------------------
def analyze_pos_embed(g, name, n_patches, use_cls):
    """Analyze positional embedding gradient for position-specific signal."""
    pe = g.get("pos_embed")
    if pe is None:
        print(f"  {name}: no pos_embed gradient")
        return None

    pe = pe.float().squeeze(0)  # (n_tokens, dim) or (n_patches + 1, dim) if cls
    print(f"  Pos embed shape: {tuple(pe.shape)}")

    # Per-position gradient norm (indicates how much signal each position carries)
    pos_norms = pe.norm(dim=1)

    if use_cls:
        cls_norm = float(pos_norms[0])
        patch_norms = pos_norms[1:]
        print(f"  CLS token grad norm: {cls_norm:.4f}")
        print(f"  Patch pos norms: min={float(patch_norms.min()):.4f}, "
              f"max={float(patch_norms.max()):.4f}, "
              f"mean={float(patch_norms.mean()):.4f}")
    else:
        patch_norms = pos_norms
        print(f"  Patch pos norms: min={float(patch_norms.min()):.4f}, "
              f"max={float(patch_norms.max()):.4f}")

    return {
        "pos_embed": pe,
        "pos_norms": pos_norms,
        "patch_norms": patch_norms,
        "use_cls": use_cls,
    }


# ---------------------------------------------------------------------------
# Patch-level reconstruction via SVD basis
# ---------------------------------------------------------------------------
def reconstruct_patches_svd(embed_info, pos_info, name, out_dir,
                            n_images=128, grid_size=None):
    """Reconstruct face-patch mosaics using the SVD basis of the patch embedding.

    Strategy: The top singular vectors of gW define the "patch vocabulary" —
    the basis patches that appear across all 128 images × all positions.
    We can reconstruct approximate patches by:
    1. Using the ratio gW/gb as direct patch estimates (noisy but real signal)
    2. Projecting these onto the top-K SVD subspace for denoising
    3. Creating full-image mosaics from the best patches
    """
    dim = embed_info["dim"]
    ph = embed_info["patch_h"]
    pw = embed_info["patch_w"]
    Vh = embed_info["Vh"]  # (min(dim, patch_pixels), patch_pixels)
    ratio_patches = embed_info["ratio_patches"]
    reliable = embed_info["reliable"]

    if grid_size is None:
        grid_size = 64 // ph  # patches per side

    n_patches = grid_size * grid_size

    # Step 1: Get ratio patches (noisy per-neuron reconstructions)
    if ratio_patches is None or reliable is None:
        print(f"  {name}: no ratio patches available, using SVD basis only")
        # Use SVD basis directly — create patch composites from singular vectors
        return _svd_basis_warmstart(embed_info, pos_info, n_images,
                                    grid_size, out_dir, name)

    # Step 2: SVD-project the ratio patches for denoising
    ratio_flat = ratio_patches.reshape(dim, -1).float()  # (dim, patch_pixels)
    k_sub = min(embed_info["eff_rank"], dim)

    # Project onto top-K subspace
    nan_mask = torch.isnan(ratio_flat)
    ratio_safe = torch.nan_to_num(ratio_flat, nan=0.0)
    Vk = Vh[:k_sub]  # (k, patch_pixels)
    coeffs = ratio_safe @ Vk.T
    projected = coeffs @ Vk
    projected_patches = projected.reshape(dim, 3, ph, pw)

    # Step 3: Quality-filter projected patches
    proj_clamped = projected_patches.clamp(0, 1)
    good_mask = reliable & torch.isfinite(projected).all(dim=1)
    proj_rng = ((projected >= -0.1) & (projected <= 1.1)).float().mean(dim=1)
    good_mask = good_mask & (proj_rng > 0.5)
    good_idx = torch.where(good_mask)[0]

    print(f"  Good projected patches: {len(good_idx)}/{dim}")

    if len(good_idx) < 4:
        print(f"  {name}: too few good patches, using SVD basis warmstart")
        return _svd_basis_warmstart(embed_info, pos_info, n_images,
                                    grid_size, out_dir, name)

    good_patches = proj_clamped[good_idx]  # (n_good, 3, ph, pw)

    # Save good patches grid
    patches_up = F.interpolate(good_patches[:64], size=(64, 64), mode="nearest")
    U.save_image_grid(patches_up,
                      os.path.join(out_dir, "good_ratio_patches.png"), ncol=8)

    # Step 4: Cluster the good patches to find distinct face regions
    feats = good_patches.reshape(len(good_idx), -1).numpy().astype(np.float64)
    from sklearn.cluster import KMeans
    n_clust = min(n_patches, len(good_idx) // 2, 64)
    if n_clust < 2:
        n_clust = min(len(good_idx), 4)
    km = KMeans(n_clusters=n_clust, n_init=4, random_state=0).fit(feats)

    # Step 5: Build mosaic images by assigning patch clusters to positions
    # Use position embedding to guide assignment
    images = _assemble_mosaics(good_patches, km.labels_, n_clust,
                               pos_info, n_images, grid_size, ph, pw)

    # Step 6: Quality filter the assembled images
    imgs_quality = U.image_stats(images)
    quality = imgs_quality["rng_frac"] * torch.clamp(imgs_quality["std"] / 0.15, max=1.0)
    quality = torch.where(imgs_quality["finite"], quality,
                          torch.full_like(quality, -1e9))
    order = torch.argsort(quality, descending=True)

    # Take top candidates
    n_keep = min(256, images.shape[0])
    images = images[order[:n_keep]]

    # Save
    U.save_image_grid(images[:64], os.path.join(out_dir, "mosaic_candidates.png"),
                      ncol=8)

    return images


def _assemble_mosaics(patches, cluster_labels, n_clusters,
                      pos_info, n_images, grid_size, ph, pw):
    """Assemble full images from patch clusters using diverse combinations."""
    n_patches = grid_size * grid_size

    # For each cluster, compute the representative patch (mean)
    cluster_reps = []
    for c in range(n_clusters):
        mask = cluster_labels == c
        if mask.sum() > 0:
            cluster_reps.append(patches[mask].mean(0))
        else:
            cluster_reps.append(torch.rand(3, ph, pw) * 0.5 + 0.25)
    cluster_reps = torch.stack(cluster_reps)  # (n_clusters, 3, ph, pw)

    # Position embedding gives us per-position signal strength
    # Use it to weight which clusters are most likely at each position
    if pos_info is not None:
        patch_norms = pos_info["patch_norms"]
        # Normalize to a weight per position
        pos_weights = patch_norms / (patch_norms.sum() + 1e-8)
    else:
        pos_weights = torch.ones(n_patches) / n_patches

    # Strategy: generate diverse images by sampling different cluster assignments
    # For each image, randomly assign clusters to positions, weighted by
    # position norms and cluster quality
    images = []
    for img_idx in range(n_images * 2):  # oversample, then quality-filter
        torch.manual_seed(img_idx * 7 + 42)
        # Random assignment with some structure:
        # - Central patches get higher-quality clusters
        # - Edge patches get more varied clusters
        grid = torch.zeros(3, grid_size * ph, grid_size * pw)

        for pos in range(n_patches):
            r, c = divmod(pos, grid_size)
            # Pick a cluster (with some randomness)
            c_idx = torch.randint(n_clusters, (1,)).item()
            patch = cluster_reps[c_idx]
            y0 = r * ph
            x0 = c * pw
            grid[:, y0:y0+ph, x0:x0+pw] = patch

        # Resize to 64x64 if needed
        if grid.shape[1] != 64 or grid.shape[2] != 64:
            grid = F.interpolate(grid.unsqueeze(0), size=(64, 64),
                                 mode="bilinear", align_corners=False).squeeze(0)

        images.append(grid.clamp(0, 1))

    return torch.stack(images)


def _svd_basis_warmstart(embed_info, pos_info, n_images, grid_size, out_dir, name):
    """Create warm-start images from SVD basis patches.

    Even without usable ratio patches, the top singular vectors of the
    patch-embedding gradient define a basis for the patch space. Random
    (but constrained) combinations of these basis patches produce
    face-structured starting points.
    """
    ph = embed_info["patch_h"]
    pw = embed_info["patch_w"]
    Vh = embed_info["Vh"]
    k = min(Vh.shape[0], embed_info["eff_rank"], 64)
    n_patches = grid_size * grid_size

    # Top-k singular vectors as patches
    basis_patches = Vh[:k].reshape(k, 3, ph, pw)
    # Normalize each to [0, 1]
    basis_patches = U.per_image_minmax(basis_patches)

    print(f"  SVD basis warmstart: {k} basis patches, "
          f"grid={grid_size}x{grid_size}")

    # Generate images by random linear combinations of basis patches
    images = []
    for img_idx in range(n_images * 2):
        torch.manual_seed(img_idx * 13 + 7)
        grid = torch.zeros(3, grid_size * ph, grid_size * pw)

        for pos in range(n_patches):
            r, c = divmod(pos, grid_size)
            # Random combination of top basis patches
            weights = torch.randn(min(k, 8)) * 0.3
            weights[0] = 1.0  # bias toward the mean patch (first singular vector)
            patch = torch.zeros(3, ph, pw)
            for j, w in enumerate(weights):
                patch += w * basis_patches[j]
            patch = patch.clamp(0, 1)

            y0 = r * ph
            x0 = c * pw
            grid[:, y0:y0+ph, x0:x0+pw] = patch

        if grid.shape[1] != 64 or grid.shape[2] != 64:
            grid = F.interpolate(grid.unsqueeze(0), size=(64, 64),
                                 mode="bilinear", align_corners=False).squeeze(0)

        images.append(grid.clamp(0, 1))

    images = torch.stack(images)

    # Quality filter
    st = U.image_stats(images)
    quality = st["rng_frac"] * torch.clamp(st["std"] / 0.15, max=1.0)
    order = torch.argsort(quality, descending=True)
    images = images[order[:256]]

    U.save_image_grid(images[:64], os.path.join(out_dir, "svd_basis_warmstart.png"),
                      ncol=8)

    return images


# ---------------------------------------------------------------------------
# Select 128 from the reconstructed images
# ---------------------------------------------------------------------------
def select_diverse_128(images, n=128):
    """Quality-ranked + greedy-dedup selection of 128 diverse images."""
    import postprocess as PP
    sel, chosen = PP.select_128(images, n=n, denoise=True)
    return sel


# ---------------------------------------------------------------------------
# Main per-model pipeline
# ---------------------------------------------------------------------------
def run_model(name, max_k=128):
    grad = U.load_grad(name)
    state = U.load_state(name)
    g = grad["gradients"]
    fam = grad.get("family")

    if fam != "vit":
        print(f"{name}: not a ViT (family={fam}), skip")
        return None

    out_dir = U.work_path("candidates_a1", name)
    U.ensure_dir(out_dir)

    print(f"\n{'='*60}")
    print(f"{name} (ViT analytic)")
    print(f"{'='*60}")

    # Determine ViT config
    if name == "model9":
        use_cls = True
        patch = 8
    elif name == "model11":
        use_cls = False
        patch = 16
    else:
        # Try to infer from gradient keys
        pw = g.get("patch_embed.proj.weight")
        if pw is not None:
            patch = pw.shape[-1]
        else:
            print(f"  {name}: can't determine ViT config")
            return None
        use_cls = "cls_token" in g

    grid_size = 64 // patch

    # Step 1: Analyze patch embedding
    embed_info = analyze_patch_embed(g, name, out_dir)
    if embed_info is None:
        return None

    # Step 2: Analyze position embedding
    n_patches = grid_size * grid_size
    pos_info = analyze_pos_embed(g, name, n_patches, use_cls)

    # Step 3: Reconstruct patches and assemble images
    images = reconstruct_patches_svd(embed_info, pos_info, name, out_dir,
                                      n_images=max_k, grid_size=grid_size)

    if images is None or images.shape[0] == 0:
        print(f"{name}: no reconstructions produced")
        return None

    # Step 4: Merge with face bank if available
    bank_path = U.work_path("outputs", "face_bank.pt")
    if os.path.exists(bank_path):
        bank = torch.load(bank_path, weights_only=False)["images"].float().clamp(0, 1)
        pool = torch.cat([images, bank], 0)
        print(f"  Merged {images.shape[0]} reconstructions + {bank.shape[0]} bank faces")
    else:
        pool = images

    # Step 5: Select 128 diverse images
    sel = select_diverse_128(pool, n=max_k)

    # Determine how many came from our reconstruction vs bank
    n_from_recon = images.shape[0]
    n_from_a1 = min(n_from_recon, max_k)

    # Step 6: Save
    torch.save({"images": sel, "source": "vit_analytic_a1",
                "n_reconstructed": n_from_recon},
               os.path.join(out_dir, "vit_selected_128.pt"))
    U.save_image_grid(sel, os.path.join(out_dir, "grid_vit_selected.png"), ncol=16)

    meta = {
        "name": name, "family": "vit",
        "patch": patch, "grid_size": grid_size, "use_cls": use_cls,
        "embed_dim": embed_info["dim"],
        "embed_eff_rank": embed_info["eff_rank"],
        "n_reconstructed": n_from_recon,
        "selected": int(sel.shape[0]),
    }
    U.save_json(meta, os.path.join(out_dir, "vit_metadata.json"))
    print(f"\n{name}: reconstructed {n_from_recon} candidates -> "
          f"selected {sel.shape[0]} -> {out_dir}")
    return meta


# ---------------------------------------------------------------------------
# Deeper ViT analytic: per-block attention gradient analysis
# ---------------------------------------------------------------------------
def analyze_deeper_layers(g, name, out_dir):
    """Analyze deeper ViT layers for additional reconstruction signal.

    The attention layers' qkv gradients encode how tokens interact.
    While the deep layers are too mixed for direct image recovery,
    they can help with:
    1. Confirming which patches belong to the same image
    2. Providing consistency constraints for jigsaw assembly
    """
    info = {}
    for key in sorted(g.keys()):
        if "attn.qkv" in key and key.endswith(".weight"):
            v = g[key].float()
            # Economy SVD
            U_, S, Vh = torch.linalg.svd(v.reshape(v.shape[0], -1),
                                          full_matrices=False)
            eff_rank = int((S > 0.01 * S[0]).sum().item())
            info[key] = {
                "shape": tuple(v.shape),
                "eff_rank": eff_rank,
                "norm": round(float(v.norm()), 4),
            }
            print(f"  {key}: shape={tuple(v.shape)}, eff_rank={eff_rank}")

    if info:
        U.save_json(info, os.path.join(out_dir, "deeper_layers.json"))
    return info


def main():
    names = sys.argv[1:] or ["model9", "model11"]
    summary = {}
    for name in names:
        try:
            summary[name] = run_model(name)
        except Exception as e:
            import traceback
            print(f"{name}: ERROR {e}")
            traceback.print_exc()
    U.save_json(summary, U.work_path("candidates_a1", "_vit_summary.json"))


if __name__ == "__main__":
    main()
