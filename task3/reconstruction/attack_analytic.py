"""Stage 2: analytic reconstruction from the first biased linear layer.

For a biased linear layer  y = W x + b  the per-sample gradient gives
    gW[row] = sum_n c_{n,row} x_n ,   gb[row] = sum_n c_{n,row}
so  gW[row] / gb[row]  is a convex/affine combination of the layer's inputs.
Rows dominated by a single sample yield a clean copy of that input.

  * MLP  (net.0, in_dim = 12288): the input IS the flattened image  -> (3,64,64).
  * CNN  (fc1): the input is act(conv(x)) flattened -> feature map; we save the
    feature and a quick RGB view, and (for full-res 64x64 features) invert the
    conv to an image in attack_conv_invert.py.

Login-node safe (no GPU, pure tensor algebra + a bounded per-model loop).
"""
import os
import torch
import torch.nn.functional as F

import utils as U


def first_leaky_linear(name, state, g):
    """Pick the analytic target: the EARLIEST biased 2-D linear layer in module
    order (net.0 for MLP, fc1 for CNN). The input to this layer is closest to
    the raw image; later linears (e.g. the classifier head) leak labels, not
    images, so fan-in size is the wrong selector."""
    for pname, w in state.items():
        if pname.endswith(".weight") and w.ndim == 2:
            base = pname[:-7]
            bkey = base + ".bias"
            if bkey in g and g[bkey] is not None:
                return base
    return None


def ratio_candidates(gW, gb, eps=1e-8):
    """candidate[row] = gW[row] / gb[row], guarded against tiny/zero bias."""
    gb = gb.float()
    gW = gW.float()
    reliable = gb.abs() > eps
    denom = torch.where(gb.abs() > eps, gb, torch.full_like(gb, float("nan")))
    ratio = gW / denom.unsqueeze(1)
    return ratio, reliable


def feature_to_rgb_view(feat, out_hw=64):
    """Cheap visual view of a feature batch (n,C,H,W) -> (n,3,64,64) in [0,1].
    Uses first 3 channels (or channel mean), per-image min-max normalised."""
    n, C, H, W = feat.shape
    if C >= 3:
        view = feat[:, :3]
    else:
        m = feat.mean(dim=1, keepdim=True)
        view = m.repeat(1, 3, 1, 1)
    if (H, W) != (out_hw, out_hw):
        view = F.interpolate(view, size=(out_hw, out_hw), mode="bilinear",
                             align_corners=False)
    return U.per_image_minmax(view)


def score_rows(cand_images, reliable):
    """Rank rows by image-likeness. cand_images: (n,3,64,64) raw ratio."""
    st = U.image_stats(cand_images)
    # in-range + contrast, only for reliable finite rows
    score = st["rng_frac"] * torch.clamp(st["std"] / 0.15, max=1.0)
    score = torch.where(reliable & st["finite"], score, torch.zeros_like(score))
    return score, st


def run_model(name, topk=256, grid_n=64):
    grad = U.load_grad(name)
    state = U.load_state(name)
    g = grad["gradients"]
    family = grad.get("family")
    feat_shape = tuple(grad.get("feature_shape"))
    out_dir = U.work_path("candidates", name)
    U.ensure_dir(out_dir)

    if family == "vit":
        # ViT: patch_embed analytic is heavily mixed; skip here, handled later.
        U.save_json({"name": name, "family": family, "note": "vit - analytic skipped"},
                    os.path.join(out_dir, "metadata.json"))
        print(f"{name}: vit, analytic skipped")
        return

    target = first_leaky_linear(name, state, g)
    if target is None:
        print(f"{name}: no biased linear target")
        return
    gW = g[target + ".weight"]
    gb = g[target + ".bias"]
    ratio, reliable = ratio_candidates(gW, gb)
    rows, in_dim = ratio.shape

    is_image_layer = in_dim == U.IMG_NUMEL
    if is_image_layer:
        cand = ratio.reshape(rows, U.IMG_C, U.IMG_H, U.IMG_W)
        images_raw = cand
        source_kind = "direct_image"
    else:
        C, H, W = feat_shape
        if C * H * W != in_dim:
            # fall back: treat as generic vector, best-effort square-ish reshape
            print(f"{name}: fc in_dim {in_dim} != prod(feat) {C*H*W}; skipping reshape")
            return
        feat = torch.nan_to_num(ratio).reshape(rows, C, H, W)
        images_raw = feature_to_rgb_view(feat)          # for scoring/viz only
        torch.save({"feature": feat, "reliable": reliable, "target": target,
                    "feature_shape": feat_shape},
                   os.path.join(out_dir, "analytic_features.pt"))
        source_kind = "feature_view"

    score, st = score_rows(images_raw, reliable)
    order = torch.argsort(score, descending=True)
    keep = order[:topk]

    # candidate bank: clamped [0,1] images for the top rows
    bank = images_raw[keep].clamp(0, 1).float()
    bank_norm = U.per_image_minmax(images_raw[keep].float())
    torch.save({
        "images": bank, "images_norm": bank_norm,
        "rows": keep, "score": score[keep],
        "source": source_kind, "target": target,
    }, os.path.join(out_dir, "analytic_candidates.pt"))

    # grids
    U.save_image_grid(bank[:grid_n], os.path.join(out_dir, "grid_top_clamp.png"))
    U.save_image_grid(bank_norm[:grid_n], os.path.join(out_dir, "grid_top_norm.png"))

    meta = {
        "name": name, "family": family, "target_layer": target,
        "in_dim": int(in_dim), "is_image_layer": bool(is_image_layer),
        "source_kind": source_kind, "feature_shape": feat_shape,
        "reliable_rows": int(reliable.sum()),
        "kept": int(keep.numel()),
        "score_top10": [float(s) for s in score[keep][:10]],
        "top_row_stats": {
            "rng_frac": [float(x) for x in st["rng_frac"][keep][:10]],
            "std": [float(x) for x in st["std"][keep][:10]],
            "tv": [float(x) for x in st["tv"][keep][:10]],
        },
    }
    U.save_json(meta, os.path.join(out_dir, "metadata.json"))
    print(f"{name}: target={target} in_dim={in_dim} kept={keep.numel()} "
          f"score[max]={float(score.max()):.3f} -> {out_dir}")


def main():
    import sys
    names = sys.argv[1:] or U.MODELS
    for name in names:
        try:
            run_model(name)
        except Exception as e:
            import traceback
            print(f"{name}: ERROR {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
