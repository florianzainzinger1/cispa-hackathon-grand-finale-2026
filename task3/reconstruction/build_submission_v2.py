"""Build a v2 submission by replacing ONLY the chosen MLP models in v1.

Starts from v1 submission.pt (READ-ONLY), swaps in the specified source for the
specified models, validates, and writes to outputs_v2/. Never writes
outputs/submission.pt or submission_v1_bank.pt.

Usage:
    python build_submission_v2.py <source> model8 model1 ...
  <source> in {exact, refined}:
    exact   -> candidates_v2/modelX/mlp_exact_selected_128.pt
    refined -> candidates_v2/modelX/mlp_refined_128.pt
If no models given, replaces none (produces a copy of v1 for sanity).
"""
import os
import sys
import torch
import utils as U
from build_submission import validate_submission

SRC = {"exact": "mlp_exact_selected_128.pt", "refined": "mlp_refined_128.pt",
       "graddenoise": "graddenoise_128.pt"}


def main():
    args = sys.argv[1:]
    source = args[0] if args and args[0] in SRC else "refined"
    models = [a for a in args if a.startswith("model")]

    v1 = torch.load(U.work_path("outputs", "submission.pt"), weights_only=False)
    sub = {k: v1[k].float().clamp(0, 1).cpu() for k in U.MODELS}

    swapped = {}
    for name in models:
        p = U.work_path("candidates_v2", name, SRC[source])
        if not os.path.exists(p):
            print(f"  [skip] {name}: {p} missing"); continue
        imgs = torch.load(p, weights_only=False)["images"].float().clamp(0, 1).cpu()
        if tuple(imgs.shape) != (128, 3, 64, 64):
            print(f"  [skip] {name}: bad shape {tuple(imgs.shape)}"); continue
        sub[name] = imgs
        swapped[name] = source
    sub = {k: v.to(torch.float32).contiguous() for k, v in sub.items()}

    errs = validate_submission(sub)
    # duplicate check per model
    for m in U.MODELS:
        u = torch.unique(sub[m].reshape(128, -1), dim=0).shape[0]
        if u < 128:
            errs.append(f"{m}: only {u}/128 unique")
    if errs:
        print("VALIDATION ERRORS:", *errs, sep="\n  - ")
        sys.exit(1)

    out = U.work_path("outputs_v2", "submission_v2_mlp.pt")
    U.ensure_dir(os.path.dirname(out))
    torch.save(sub, out)
    preview = torch.cat([sub[m][:12] for m in U.MODELS], 0)
    U.save_image_grid(preview, U.work_path("outputs_v2", "submission_v2_mlp_preview.png"), ncol=12)
    stats = {m: {"swapped": swapped.get(m, "v1"),
                 "min": float(sub[m].min()), "max": float(sub[m].max()),
                 "mean": float(sub[m].mean())} for m in U.MODELS}
    U.save_json(stats, U.work_path("outputs_v2", "submission_v2_mlp_stats.json"))

    # confirm v1 untouched
    v1_size = os.path.getsize(U.work_path("outputs", "submission.pt"))
    print(f"swapped: {swapped}")
    print(f"saved {out}  (v1 submission.pt still {v1_size} bytes, untouched)")
    print("VALID")


if __name__ == "__main__":
    main()
