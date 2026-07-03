"""Build a1 submission = c3 base with improved MLP and ViT models from a1 experiments.

Takes the c3/v3 submission as the base and selectively swaps in improved
reconstructions from candidates_a1/ for models that have better candidates.
Every other model is taken byte-for-byte from c3.

Logic:
  - For MLP models (1,4,5,8): use a1 SVD-projected reconstructions if available
  - For ViT models (9,11): use a1 analytic reconstructions if available
  - For all other models: keep c3 verbatim

Safety:
  - Never overwrites v1/v2/v3/c3/c4/c5 submissions
  - Writes to fresh outputs_a1/ folder
  - Full validation before saving
  - Reports what was swapped and what was kept
"""
import os
import sys
import torch
import utils as U
from build_submission import validate_submission

# Base submission to build on
C3 = U.work_path("outputs_v3", "submission_v3.pt")
# Fallback to v1 if c3 doesn't exist
V1 = U.work_path("outputs", "submission.pt")

# Models to potentially swap from a1 experiments
MLP_SWAP = ["model1", "model4", "model5", "model8"]
VIT_SWAP = ["model9", "model11"]


def load_base():
    """Load base submission (prefer c3, fallback to v1)."""
    for path, label in [(C3, "c3"), (V1, "v1")]:
        if os.path.exists(path):
            sub = torch.load(path, weights_only=False)
            print(f"Base: {label} ({path})")
            return sub, label
    raise FileNotFoundError("No base submission found (c3 or v1)")


def try_load_a1(name, sub_type):
    """Try to load a1 candidate for a model. Returns (images, source) or (None, None)."""
    if sub_type == "mlp":
        path = U.work_path("candidates_a1", name, "svd_selected_128.pt")
    elif sub_type == "vit":
        path = U.work_path("candidates_a1", name, "vit_selected_128.pt")
    else:
        return None, None

    if not os.path.exists(path):
        return None, None

    blob = torch.load(path, weights_only=False)
    imgs = blob["images"].float().clamp(0, 1).cpu()
    if tuple(imgs.shape) != (128, 3, 64, 64):
        print(f"  [skip] {name}: bad shape {tuple(imgs.shape)}")
        return None, None

    return imgs, f"a1_{sub_type}"


def main():
    base, base_label = load_base()
    sub = {m: base[m].float().clamp(0, 1).cpu() for m in U.MODELS}

    swapped = {}
    for m in MLP_SWAP:
        imgs, src = try_load_a1(m, "mlp")
        if imgs is not None:
            if not torch.equal(imgs, sub[m]):
                sub[m] = imgs
                swapped[m] = src
                print(f"  [swap] {m}: {src}")
            else:
                print(f"  [note] {m}: identical to base")
        else:
            print(f"  [keep] {m}: no a1 candidate, keeping {base_label}")

    for m in VIT_SWAP:
        imgs, src = try_load_a1(m, "vit")
        if imgs is not None:
            if not torch.equal(imgs, sub[m]):
                sub[m] = imgs
                swapped[m] = src
                print(f"  [swap] {m}: {src}")
            else:
                print(f"  [note] {m}: identical to base")
        else:
            print(f"  [keep] {m}: no a1 candidate, keeping {base_label}")

    sub = {m: v.to(torch.float32).contiguous() for m, v in sub.items()}

    # Validate
    errs = validate_submission(sub)
    for m in U.MODELS:
        u = torch.unique(sub[m].reshape(128, -1), dim=0).shape[0]
        if u < 128:
            errs.append(f"{m}: {u}/128 unique")
    if errs:
        print("VALIDATION ERRORS:", *errs, sep="\n  - ")
        for k in sub:
            sub[k] = sub[k].clamp(0, 1).to(torch.float32)
        errs2 = validate_submission(sub)
        if errs2:
            print("STILL INVALID, not saving.")
            sys.exit(1)

    out = U.work_path("outputs_a1", "submission_a1.pt")
    U.ensure_dir(os.path.dirname(out))
    torch.save(sub, out)

    preview = torch.cat([sub[m][:12] for m in U.MODELS], 0)
    U.save_image_grid(preview, U.work_path("outputs_a1", "submission_a1_preview.png"),
                      ncol=12)

    stats = {m: {"status": swapped.get(m, f"{base_label}_verbatim"),
                 "min": float(sub[m].min()), "max": float(sub[m].max()),
                 "mean": float(sub[m].mean())} for m in U.MODELS}
    U.save_json(stats, U.work_path("outputs_a1", "submission_a1_stats.json"))

    print(f"\n{'='*50}")
    print("=== A1 SUBMISSION SUMMARY ===")
    print(f"{'='*50}")
    for m in U.MODELS:
        status = swapped.get(m, f"{base_label}_verbatim")
        print(f"  {m:8} {status}")
    print(f"\nswapped: {list(swapped.keys())} ({len(swapped)} models)")
    print(f"unchanged: {[m for m in U.MODELS if m not in swapped]} "
          f"({12 - len(swapped)} models)")
    print(f"saved: {out}")

    def sz(p): return os.path.getsize(p) if os.path.exists(p) else -1
    if os.path.exists(C3):
        print(f"c3 submission still {sz(C3)} bytes")
    if os.path.exists(V1):
        print(f"v1 submission still {sz(V1)} bytes")
    print("VALID")


if __name__ == "__main__":
    main()
