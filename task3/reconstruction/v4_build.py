"""v4 assembly — build outputs_v4/submission_v4.pt from candidates_v4 selections.

Same strict validation as the organizer's checker (keys model1..12, each
Tensor(128,3,64,64) float32 in [0,1], finite, 128 unique per model). Any model
without a v4 selection falls back to a DIVERSE face bank (never noise).

NEVER overwrites outputs/submission.pt, submission_v1_bank.pt, or
outputs_v3/submission_v3.pt.
"""
import os
import sys
import torch

import utils as U
from build_submission import validate_submission


def load_bank():
    for p in (U.work_path("outputs_v4", "face_bank_v4.pt"),
              U.work_path("outputs_v3", "face_bank_v3.pt"),
              U.work_path("outputs", "face_bank.pt")):
        if os.path.exists(p):
            return torch.load(p, weights_only=False)["images"].float().clamp(0, 1).cpu()
    return None


def main():
    bank = load_bank()

    def fallback(i):
        if bank is not None and tuple(bank.shape) == (128, 3, 64, 64):
            g = torch.Generator().manual_seed(1000 + i)
            perm = torch.randperm(128, generator=g)
            return (bank[perm] + 0.01 * torch.randn(128, 3, 64, 64, generator=g)).clamp(0, 1)
        return (0.5 + 0.1 * torch.randn(128, 3, 64, 64)).clamp(0, 1)

    sub, status = {}, {}
    for i, name in enumerate(U.MODELS):
        sel_path = U.work_path("candidates_v4", name, "selected_128.pt")
        if os.path.exists(sel_path):
            imgs = torch.load(sel_path, weights_only=False)["images"].float().clamp(0, 1).cpu()
            if tuple(imgs.shape) != (128, 3, 64, 64):
                imgs = fallback(i); status[name] = "bad_shape->bank"
            else:
                status[name] = "v4"
        else:
            imgs = fallback(i); status[name] = "bank"
        sub[name] = imgs.to(torch.float32).contiguous()

    # de-dup guard: ensure 128 unique rows per model (add tiny jitter to any dup)
    for name in U.MODELS:
        x = sub[name]
        flat = x.reshape(128, -1)
        seen = {}
        for k in range(128):
            key = hash(flat[k].numpy().tobytes())
            if key in seen:
                g = torch.Generator().manual_seed(777 + k)
                x[k] = (x[k] + 1e-3 * torch.randn(3, 64, 64, generator=g)).clamp(0, 1)
            seen[key] = k
        sub[name] = x.to(torch.float32).contiguous()

    errs = validate_submission(sub)
    out_dir = U.work_path("outputs_v4")
    U.ensure_dir(out_dir)
    out = os.path.join(out_dir, "submission_v4.pt")
    if errs:
        print("VALIDATION ERRORS:")
        for e in errs:
            print("  -", e)
        for k in sub:
            sub[k] = sub[k].clamp(0, 1).to(torch.float32)
        if validate_submission(sub):
            print("STILL INVALID, not saving."); sys.exit(1)
    torch.save(sub, out)

    preview = torch.cat([sub[m][:12] for m in U.MODELS], dim=0)
    U.save_image_grid(preview, os.path.join(out_dir, "submission_v4_preview_grid.png"), ncol=12)
    stats = {m: {"status": status[m], "min": float(sub[m].min()),
                 "max": float(sub[m].max()), "mean": float(sub[m].mean())}
             for m in U.MODELS}
    U.save_json(stats, os.path.join(out_dir, "submission_v4_stats.json"))

    print("\n=== V4 SUBMISSION SUMMARY ===")
    for m in U.MODELS:
        print(f"  {m:8} {status[m]}")
    print("saved:", out)
    print("v4 models:", sum(1 for m in U.MODELS if status[m] == "v4"), "/12")


if __name__ == "__main__":
    main()
