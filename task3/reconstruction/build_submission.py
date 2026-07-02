"""Stage 6: assemble outputs/submission.pt and validate it exactly like the
organizer's checker (keys model1..model12, shape (128,3,64,64), float, [0,1],
finite). Missing models are filled with a neutral placeholder so the file is
always valid; a warning lists which models are placeholders.
"""
import os
import sys
import torch

import utils as U


def validate_submission(sub):
    errs = []
    expected = set(U.MODELS)
    got = set(sub.keys())
    if got != expected:
        errs.append(f"keys mismatch: missing={expected-got} extra={got-expected}")
    for k in sorted(got & expected):
        t = sub[k]
        if not torch.is_tensor(t):
            errs.append(f"{k}: not a tensor"); continue
        if tuple(t.shape) != (128, 3, 64, 64):
            errs.append(f"{k}: shape {tuple(t.shape)} != (128,3,64,64)")
        if not torch.is_floating_point(t):
            errs.append(f"{k}: dtype {t.dtype} not float")
        if not torch.isfinite(t).all():
            errs.append(f"{k}: has NaN/Inf")
        mn, mx = float(t.min()), float(t.max())
        if mn < 0 or mx > 1:
            errs.append(f"{k}: range [{mn:.3f},{mx:.3f}] outside [0,1]")
    return errs


def placeholder(seed=0):
    g = torch.Generator().manual_seed(seed)
    # mild grey noise - valid, low-score fallback until a real attack lands
    return (0.5 + 0.1 * torch.randn(128, 3, 64, 64, generator=g)).clamp(0, 1)


def main():
    sub = {}
    status = {}
    for i, name in enumerate(U.MODELS):
        sel_path = U.work_path("candidates", name, "selected_128.pt")
        if os.path.exists(sel_path):
            imgs = torch.load(sel_path, weights_only=False)["images"].float()
            imgs = imgs.clamp(0, 1).cpu()
            if tuple(imgs.shape) != (128, 3, 64, 64):
                imgs = placeholder(i); status[name] = "BAD_SHAPE->placeholder"
            else:
                status[name] = "real"
        else:
            imgs = placeholder(i); status[name] = "placeholder"
        sub[name] = imgs.to(torch.float32).contiguous()

    errs = validate_submission(sub)
    out = U.work_path("outputs", "submission.pt")
    U.ensure_dir(os.path.dirname(out))
    if errs:
        print("VALIDATION ERRORS:")
        for e in errs:
            print("  -", e)
        # still try to fix ranges by clamping
        for k in sub:
            sub[k] = sub[k].clamp(0, 1).to(torch.float32)
        errs2 = validate_submission(sub)
        if errs2:
            print("STILL INVALID, not saving.")
            sys.exit(1)
    torch.save(sub, out)

    # preview: one row (first 12 imgs) per model stacked
    preview = torch.cat([sub[m][:12] for m in U.MODELS], dim=0)
    U.save_image_grid(preview, U.work_path("outputs", "submission_preview_grid.png"),
                      ncol=12)
    stats = {m: {"status": status[m],
                 "min": float(sub[m].min()), "max": float(sub[m].max()),
                 "mean": float(sub[m].mean())} for m in U.MODELS}
    U.save_json(stats, U.work_path("outputs", "submission_stats.json"))

    print("\n=== SUBMISSION SUMMARY ===")
    for m in U.MODELS:
        print(f"  {m:8} {status[m]}")
    print("saved:", out)
    real = sum(1 for m in U.MODELS if status[m] == "real")
    print(f"real models: {real}/12")


if __name__ == "__main__":
    main()
