"""Build c4 = c3 with ONLY the strided CNNs (2,3,7) swapped to the better
re-inversion. Every other model is taken byte-for-byte from c3 (submission_v3.pt),
so c4 can only differ from c3 on 2/3/7. Writes a fresh outputs_c4/submission_c4.pt;
never touches v1/v2/v3/v4 files.
"""
import os
import torch
import utils as U
from build_submission import validate_submission

C3 = U.work_path("outputs_v3", "submission_v3.pt")
SWAP = ["model2", "model3", "model7"]


def main():
    c3 = torch.load(C3, weights_only=False)
    sub = {m: c3[m].float().clamp(0, 1).cpu() for m in U.MODELS}

    swapped = {}
    for m in SWAP:
        p = U.work_path("candidates_v4", m, "selected_128.pt")
        if not os.path.exists(p):
            print(f"  [skip] {m}: {p} missing — keeping c3"); continue
        imgs = torch.load(p, weights_only=False)["images"].float().clamp(0, 1).cpu()
        if tuple(imgs.shape) != (128, 3, 64, 64):
            print(f"  [skip] {m}: bad shape {tuple(imgs.shape)} — keeping c3"); continue
        # only swap if it actually changed vs c3
        if torch.equal(imgs, sub[m]):
            print(f"  [note] {m}: identical to c3"); continue
        sub[m] = imgs
        swapped[m] = "v4_strided"
    sub = {m: v.to(torch.float32).contiguous() for m, v in sub.items()}

    errs = validate_submission(sub)
    for m in U.MODELS:
        u = torch.unique(sub[m].reshape(128, -1), dim=0).shape[0]
        if u < 128:
            errs.append(f"{m}: {u}/128 unique")
    if errs:
        print("VALIDATION ERRORS:", *errs, sep="\n  - "); raise SystemExit(1)

    out = U.work_path("outputs_c4", "submission_c4.pt")
    U.ensure_dir(os.path.dirname(out))
    torch.save(sub, out)
    preview = torch.cat([sub[m][:12] for m in U.MODELS], 0)
    U.save_image_grid(preview, U.work_path("outputs_c4", "submission_c4_preview.png"), ncol=12)

    # confirm untouched baselines
    def sz(p): return os.path.getsize(p) if os.path.exists(p) else -1
    print("swapped:", swapped)
    print("unchanged-from-c3 models:", [m for m in U.MODELS if m not in swapped])
    print(f"saved {out}")
    print(f"c3 submission_v3.pt still {sz(C3)} bytes; "
          f"v1 submission.pt still {sz(U.work_path('outputs','submission.pt'))} bytes")
    print("VALID")


if __name__ == "__main__":
    main()
