"""Subspace-projection denoising.

The first-layer weight-gradient is gW = sum_{n=1..128} c_n x_n^T + noise, so its
top-128 right singular vectors span the 128 true inputs; the tail (128..1024) is
a noise floor (verified: 2.6% for clean MLPs, 5.8% model8, 8.3% model6). The
per-neuron ratio candidate r_i = gW[i]/gB[i] carries that same noise. Projecting
each r_i onto the top-K singular subspace removes the off-signal noise while
keeping the face content.

  * MLP  (net.0): r_i is a (3,64,64) image -> project in image space, save grids.
  * CNN  (fc1):   r_i is the (C,H,W) fc1-input feature -> project in feature
    space, save projected features for a later conv-inversion (GPU).

Login-node safe (SVD of <=1024x32768 + projection; no optimization).
"""
import os
import sys
import torch

import utils as U


def first_biased_linear(state, g):
    for pname, w in state.items():
        if pname.endswith(".weight") and w.ndim == 2:
            base = pname[:-7]
            if base + ".bias" in g and g[base + ".bias"] is not None:
                return base
    return None


def subspace_basis(gW, k):
    """Top-k right singular vectors V (in, k) of gW (out, in)."""
    _, S, Vh = torch.linalg.svd(gW.float(), full_matrices=False)
    k = min(k, Vh.shape[0])
    return Vh[:k].t().contiguous(), S[:k]        # V:(in,k)


def project_rows(R, V):
    """Project each row of R (n,in) onto span(V). Returns (n,in)."""
    coeff = R @ V                                # (n,k)
    return coeff @ V.t()                         # (n,in)


def run(name, k=128, topn=384):
    grad = U.load_grad(name)
    state = U.load_state(name)
    g = grad["gradients"]
    fam = grad.get("family")
    fs = grad.get("feature_shape")
    base = first_biased_linear(state, g)
    gW = g[base + ".weight"].float()
    gB = g[base + ".bias"].float()
    out_dir = U.work_path("candidates_v5", name)
    U.ensure_dir(out_dir)

    # ratio candidates (guard tiny bias)
    denom = torch.where(gB.abs() > 1e-8, gB, torch.full_like(gB, float("nan")))
    R = (gW / denom.unsqueeze(1))                # (out, in)
    finite = torch.isfinite(R).all(dim=1)
    R = R[finite]
    # rank rows by |gB| (reliability) and keep topn
    order = torch.argsort(gB[finite].abs(), descending=True)[:topn]
    R = R[order]

    V, S = subspace_basis(gW, k)
    Rp = project_rows(torch.nan_to_num(R), V)

    if gW.shape[1] == U.IMG_NUMEL:               # MLP: image space
        before = U.per_image_minmax(R.reshape(-1, U.IMG_C, U.IMG_H, U.IMG_W))
        after = U.per_image_minmax(Rp.reshape(-1, U.IMG_C, U.IMG_H, U.IMG_W))
        U.save_image_grid(before[:64], os.path.join(out_dir, f"{name}_before.png"))
        U.save_image_grid(after[:64], os.path.join(out_dir, f"{name}_after.png"))
        torch.save({"images": after.clamp(0, 1)}, os.path.join(out_dir, "proj_images.pt"))
        print(f"{name}: MLP image-space projection k={k} -> {out_dir}", flush=True)
    else:                                        # CNN: feature space
        C, H, W = fs
        feat_before = R.reshape(-1, C, H, W)
        feat_after = Rp.reshape(-1, C, H, W)
        # cheap RGB view (first 3 channels) for a quick look
        def view(f):
            v = f[:, :3] if f.shape[1] >= 3 else f.mean(1, keepdim=True).repeat(1, 3, 1, 1)
            import torch.nn.functional as F
            if (H, W) != (64, 64):
                v = F.interpolate(v, size=(64, 64), mode="bilinear", align_corners=False)
            return U.per_image_minmax(v)
        U.save_image_grid(view(feat_before)[:64], os.path.join(out_dir, f"{name}_feat_before.png"))
        U.save_image_grid(view(feat_after)[:64], os.path.join(out_dir, f"{name}_feat_after.png"))
        torch.save({"feature": feat_after, "feature_shape": fs, "k": k},
                   os.path.join(out_dir, "proj_features.pt"))
        print(f"{name}: CNN feature-space projection k={k} feat={fs} -> {out_dir}", flush=True)


def main():
    k = int(os.environ.get("FL_K", "128"))
    names = sys.argv[1:] or ["model8", "model6"]
    for name in names:
        try:
            run(name, k=k)
        except Exception as e:
            import traceback
            print(f"{name}: ERROR {e}"); traceback.print_exc()


if __name__ == "__main__":
    main()
