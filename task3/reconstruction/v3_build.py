"""v3 assembly — build outputs_v3/submission_v3.pt from candidates_v3 selections.

Same strict validation as the organizer's checker (keys model1..12, each
Tensor(128,3,64,64) float32 in [0,1], finite). Any model without a v3 selection
falls back to a DIVERSE face bank (never grey/RGB noise).
"""
import os
import sys
import torch

import utils as U
from build_submission import validate_submission


def load_bank():
    p = U.work_path("outputs", "face_bank.pt")
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
        sel_path = U.work_path("candidates_v3", name, "selected_128.pt")
        if os.path.exists(sel_path):
            imgs = torch.load(sel_path, weights_only=False)["images"].float().clamp(0, 1).cpu()
            if tuple(imgs.shape) != (128, 3, 64, 64):
                imgs = fallback(i); status[name] = "bad_shape->bank"
            else:
                status[name] = "v3"
        else:
            imgs = fallback(i); status[name] = "bank"
        sub[name] = imgs.to(torch.float32).contiguous()

    errs = validate_submission(sub)
    out_dir = U.work_path("outputs_v3")
    U.ensure_dir(out_dir)
    out = os.path.join(out_dir, "submission_v3.pt")
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
    U.save_image_grid(preview, os.path.join(out_dir, "submission_v3_preview_grid.png"), ncol=12)
    stats = {m: {"status": status[m], "min": float(sub[m].min()),
                 "max": float(sub[m].max()), "mean": float(sub[m].mean())}
             for m in U.MODELS}
    U.save_json(stats, os.path.join(out_dir, "submission_v3_stats.json"))

    print("\n=== V3 SUBMISSION SUMMARY ===")
    for m in U.MODELS:
        print(f"  {m:8} {status[m]}")
    print("saved:", out)
    print("v3 models:", sum(1 for m in U.MODELS if status[m] == "v3"), "/12")


if __name__ == "__main__":
    main()
