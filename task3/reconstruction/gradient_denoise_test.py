"""Test: low-rank-denoise gW (remove out-of-signal-subspace noise) BEFORE taking
analytic ratios. gW's signal lies in its top-~128 singular directions; the rest
is perturbation noise. Project gW to rank k, then ratio = gW_lr/gb, and select
128. Compare span-capture and visuals vs v1. READ-ONLY; writes candidates_v2/.
"""
import os
import sys
import numpy as np
import torch
import utils as U
import compare_fidelity as CF
import postprocess as PP


def run(name, k=128):
    grad = U.load_grad(name)
    state = U.load_state(name)
    g = grad["gradients"]
    base = CF.first_image_linear(state, g)
    gW = g[base + ".weight"].float().numpy()
    gb = g[base + ".bias"].float().numpy()

    Uu, S, Vt = np.linalg.svd(gW, full_matrices=False)
    gW_lr = (Uu[:, :k] * S[:k]) @ Vt[:k]                 # rank-k denoised

    reliable = np.abs(gb) > 1e-8
    ratio = np.zeros_like(gW_lr)
    ratio[reliable] = gW_lr[reliable] / gb[reliable, None]
    imgs = torch.from_numpy(ratio).float().reshape(-1, 3, 64, 64)
    st = U.image_stats(imgs)
    keep = reliable & st["finite"].numpy() & (st["rng_frac"].numpy() > 0.6) & (st["std"].numpy() > 0.02)
    pool = imgs[torch.from_numpy(keep)].clamp(0, 1)
    if pool.shape[0] < 8:
        print(f"{name}: only {pool.shape[0]} valid denoised rows"); return None
    sel, _ = PP.select_128(pool, n=128, denoise=True)
    sel = sel.float().clamp(0, 1)

    cap_new, _ = CF.captured(gW, sel.reshape(128, -1).numpy())
    v1 = CF.load_imgs(U.work_path("candidates", name, "selected_128.pt"))
    cap_v1, _ = CF.captured(gW, v1)

    out_dir = U.work_path("candidates_v2", name)
    U.ensure_dir(out_dir)
    torch.save({"images": sel}, os.path.join(out_dir, "graddenoise_128.pt"))
    U.save_image_grid(sel, os.path.join(out_dir, "graddenoise_grid.png"), ncol=16)
    v1t = torch.from_numpy(v1).float().reshape(128, 3, 64, 64)
    U.save_image_grid(torch.cat([v1t[:32], sel[:32]], 0),
                      os.path.join(out_dir, "compare_v1_vs_graddenoise.png"), ncol=16)
    print(f"{name}: valid={pool.shape[0]} capture v1={cap_v1:.4f} graddenoise={cap_new:.4f} "
          f"(delta {cap_new-cap_v1:+.4f})")
    return {"name": name, "cap_v1": cap_v1, "cap_graddenoise": cap_new}


def main():
    names = sys.argv[1:] or ["model8", "model1", "model4", "model5"]
    res = [run(n) for n in names]
    U.save_json([r for r in res if r], U.work_path("reports", "graddenoise_capture.json"))


if __name__ == "__main__":
    main()
