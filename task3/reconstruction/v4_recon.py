"""v4 conv inversion for the strided CNN models (2, 3, 7).

KEY FINDING (synth_proxy.py, Hungarian one-to-one SSIM on real faces pushed
through each model's real conv, then inverted):

  the v3 pipeline optimised the low-res latent at base_res = feature_res, which
  over-constrains the image to the (very coarse) feature grid and loses facial
  structure. Optimising at ~2x the feature resolution is the sweet spot:

    model  stride feat  v3 base  v3 SSIM   v4 base  v4 SSIM   gain
    model2   4    16      16      0.592      32      0.664    +0.072
    model7   4    16      16      0.559      32      0.640    +0.081
    model3   8     8       8      0.291      16      0.423    +0.132   (v3 lost to bank!)

So v4 re-inverts these three models at the better base_res, writing a fresh
conv_invert pool to candidates_v4/<model>/conv_invert_candidates.pt.

Full-res (stride-1) models 6,10,12 and the MLP/ViT models are unchanged -> v4
selection reads their v3 candidates directly.

GPU stage: run via job_v4_recon.sh (sbatch). Env FL_DEVICE=cuda.
"""
import os
import sys
import torch

import utils as U
import models as M
from attack_conv_invert import invert_conv

# base_res per strided model, tuned by a 2-seed Hungarian-SSIM sweep
# (synth_proxy2.py). Optimum is ~1.5x feature res for stride-4 and 2x for
# stride-8; larger base_res reintroduces checkerboard, smaller loses structure.
#   model2: base=24 -> 0.675 (vs 0.592 at v3 base=16)
#   model7: base=24 -> 0.650 (vs 0.559 at v3 base=16)
#   model3: base=16 -> 0.423 (vs 0.291 at v3 base=8; also beats bank 0.34)
STRIDED_CFG = {
    "model2": dict(base_res=24, steps=800, lr=0.03, tv=2e-3),
    "model7": dict(base_res=24, steps=800, lr=0.03, tv=2e-3),
    "model3": dict(base_res=16, steps=800, lr=0.03, tv=3e-3),
}


def run_model(name, device="cpu", topk=256):
    cfg = STRIDED_CFG[name]
    grad = U.load_grad(name)
    state = U.load_state(name)
    meta = {"family": "cnn", "activation": grad["activation"],
            "feature_shape": grad["feature_shape"]}
    model = M.build_model(name, meta, state)

    fd = torch.load(U.work_path("candidates", name, "analytic_features.pt"),
                    weights_only=False)
    feat = fd["feature"]                       # (1024,C,H,W) raw ratio
    cand = torch.load(U.work_path("candidates", name, "analytic_candidates.pt"),
                      weights_only=False)
    rows = cand["rows"][:topk]
    targ = feat[rows]
    act = grad["activation"]
    if act == "relu":
        targ = targ.clamp_min(0.0)
    elif act == "sigmoid":
        targ = targ.clamp(0.0, 1.0)
    elif act == "tanh":
        targ = targ.clamp(-1.0, 1.0)
    targ = torch.nan_to_num(targ)

    rec = invert_conv(model, targ, steps=cfg["steps"], lr=cfg["lr"],
                      tv=cfg["tv"], device=device, base_res=cfg["base_res"])

    out_dir = U.work_path("candidates_v4", name)
    U.ensure_dir(out_dir)
    torch.save({"images": rec, "rows": rows, "source": "conv_invert_v4",
                "base_res": cfg["base_res"]},
               os.path.join(out_dir, "conv_invert_candidates.pt"))
    U.save_image_grid(rec[:128], os.path.join(out_dir, "grid_conv_invert_v4.png"), ncol=16)
    print(f"{name}: base_res={cfg['base_res']} steps={cfg['steps']} "
          f"inverted {rec.shape[0]} -> {out_dir}", flush=True)


def main():
    device = os.environ.get("FL_DEVICE", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print("device:", device)
    names = sys.argv[1:] or ["model2", "model7", "model3"]
    for name in names:
        try:
            run_model(name, device=device)
        except Exception as e:
            import traceback
            print(f"{name}: ERROR {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
