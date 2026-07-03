"""v5: conv-invert model6's subspace-denoised fc1 features to recover far more
clean faces than v3's 46. model6's conv is stride-1 3->8 at full 64x64, so the
inversion is well-determined; v3 under-recovered because it inverted the noisy
(un-denoised) features and over-filtered. Here we invert the top-N denoised
features and keep the cleanest by TV.
"""
import os
import sys
import torch

import utils as U
import models as M
from attack_conv_invert import invert_conv


def run(name, device, steps=300):
    grad = U.load_grad(name)
    state = U.load_state(name)
    meta = {"family": "cnn", "activation": grad["activation"],
            "feature_shape": grad["feature_shape"]}
    model = M.build_model(name, meta, state)
    act = grad["activation"]

    pf = U.work_path("candidates_v5", name, "proj_features.pt")
    feat = torch.load(pf, weights_only=False)["feature"].float()   # (n,C,H,W)
    if act == "relu":
        feat = feat.clamp_min(0.0)
    elif act == "sigmoid":
        feat = feat.clamp(0.0, 1.0)
    elif act == "tanh":
        feat = feat.clamp(-1.0, 1.0)

    rec = invert_conv(model, feat, steps=steps, lr=0.05, tv=1e-4,
                      device=device, base_res=64)
    out_dir = U.work_path("candidates_v5", name)
    U.ensure_dir(out_dir)
    torch.save({"images": rec}, os.path.join(out_dir, "conv_invert.pt"))
    U.save_image_grid(rec[:96], os.path.join(out_dir, "grid_conv_invert.png"), ncol=16)

    # quick cleanliness stats
    tv = U.total_variation(rec)
    clean = int((tv < 0.20).sum())
    print(f"{name}: inverted {rec.shape[0]} feats, {clean} with TV<0.20 "
          f"(v3 kept 46) -> {out_dir}/grid_conv_invert.png", flush=True)


def main():
    device = os.environ.get("FL_DEVICE", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    steps = int(os.environ.get("FL_ITERS", "300"))
    names = sys.argv[1:] or ["model6"]
    for name in names:
        try:
            run(name, device, steps=steps)
        except Exception as e:
            import traceback
            print(f"{name}: ERROR {e}"); traceback.print_exc()


if __name__ == "__main__":
    main()
