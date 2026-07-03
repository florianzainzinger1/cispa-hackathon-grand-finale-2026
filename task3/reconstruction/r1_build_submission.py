"""Assemble an r1 experimental submission from the StyleGAN-prior candidates.

For each model1..12: use runs/r1/<model>/selected_128.pt if present (the r1
StyleGAN candidates), otherwise fall back to the best EXISTING per-model result
(v3 submission, else v1). Writes to a FRESH path under runs/r1/submissions/ --
never overwrites outputs/submission.pt, submission_v1_bank.pt, or any v1/v2/v3
file. The human decides whether to submit it.

This lets the r1 candidates for the hard models (9/11 ViT, 2 CNN, ...) be dropped
into an otherwise-best submission and compared on the leaderboard.
"""
import os
import torch

import utils as U
import build_submission as BS   # reuse validate_submission

# Candidate subdirs in PRIORITY order (first hit wins per model). r2 (W-space +
# truncation, clean faces) preferred over r1 (which has the project results for
# model3/model7 and the earlier degenerate latent runs).
R_SUBDIRS = ["runs/r2", "runs/r1"]
BASE_CANDIDATES = [
    ("outputs_v3", "submission_v3.pt"),
    ("outputs_v2", "submission_v2_mlp.pt"),
    ("outputs", "submission.pt"),
]


def load_base():
    for rel in BASE_CANDIDATES:
        p = U.work_path(*rel)
        if os.path.exists(p):
            try:
                return torch.load(p, weights_only=False), p
            except Exception as e:
                print(f"[base] failed to load {p}: {e}")
    return None, None


def load_r(name):
    """Return (images, subdir) from the first priority subdir that has a valid
    selected_128.pt for `name`, else (None, None)."""
    for sd in R_SUBDIRS:
        p = U.work_path(sd, name, "selected_128.pt")
        if not os.path.exists(p):
            continue
        t = torch.load(p, weights_only=False)["images"].float().clamp(0, 1)
        if tuple(t.shape) == (128, 3, 64, 64) and torch.isfinite(t).all():
            return t.contiguous(), sd
        print(f"[r] {name}: bad shape/finite in {sd} {tuple(t.shape)} -> skip")
    return None, None


def main():
    import sys
    # Optional: restrict which models take r-candidates (default: all present)
    only = set(sys.argv[1:]) if len(sys.argv) > 1 else None

    base, base_p = load_base()
    print(f"[base] using {base_p}")
    sub, prov = {}, {}
    for name in U.MODELS:
        imgs, sd = load_r(name) if (only is None or name in only) else (None, None)
        if imgs is not None:
            sub[name], prov[name] = imgs, f"{sd.split('/')[-1]}_stylegan"
        elif base is not None and name in base:
            sub[name] = base[name].float().clamp(0, 1).contiguous()
            prov[name] = f"base:{os.path.basename(base_p)}"
        else:
            sub[name] = torch.rand(128, 3, 64, 64)
            prov[name] = "placeholder"

    errs = BS.validate_submission(sub)
    out = U.work_path("runs", "submissions", "submission_r1r2.pt")
    U.ensure_dir(os.path.dirname(out))
    torch.save(sub, out)
    U.save_json({"provenance": prov, "base": base_p, "errors": errs},
                U.work_path("runs", "submissions", "provenance.json"))

    # preview montage: 8 faces per model (rows), models top->bottom
    rows = [sub[name][:8] for name in U.MODELS]
    U.save_image_grid(torch.cat(rows, 0),
                      U.work_path("runs", "submissions", "preview_montage.png"),
                      ncol=8)

    n_r = sum(1 for v in prov.values() if "stylegan" in v)
    print(f"[r_build] provenance: {prov}")
    print(f"[r_build] {n_r}/12 models from r1/r2 stylegan; wrote {out} "
          f"({'VALID' if not errs else 'ERRORS: ' + str(errs)})")


if __name__ == "__main__":
    main()
