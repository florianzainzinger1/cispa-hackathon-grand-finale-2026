"""Build c5 = c3 with ONLY model6 rebuilt.

Fresh analysis showed model6's fc1 gradient is a full rank-128 face signal under
an 8.3% noise floor: its conv-inversions are mostly real (grainy) faces, but v3's
TV<0.20 filter kept only 46 and bank-filled the other 82. c5 relaxes the filter,
keeps ~all recognizable grainy faces from the subspace-denoised v5 inversions,
adaptive-denoises the grain, and bank-fills only the remainder. Every other model
is byte-identical to c3, so c5 can only differ from c3 on model6.
"""
import os
import torch

import utils as U
import v3_select as S
from build_submission import validate_submission

C3 = U.work_path("outputs_v3", "submission_v3.pt")


def select_faces(imgs, bank, n=128, tv_cut=0.34, dedup_eps=0.03):
    """Keep distinct recognizable faces IN RELIABILITY ORDER (imgs are already
    sorted by |gB|). Reject only true garbage: dark/flat images (low contrast)
    and extreme-TV noise. Light denoise, then bank-fill any remainder.

    NB: we do NOT rank by the smoothness-based quality score — it prefers
    dark-smooth blanks over bright grainy faces, which is exactly wrong here.
    """
    st = U.image_stats(imgs)
    feats = S.feat_vectors(imgs)
    # face gate: finite, has contrast, not too dark/bright, not pure noise
    ok = (st["finite"] & (st["std"] > 0.08) & (st["tv"] < tv_cut)
          & (imgs.mean(dim=(1, 2, 3)) > 0.12) & (imgs.mean(dim=(1, 2, 3)) < 0.9))
    reps, chosen_feats = [], []
    for i in range(imgs.shape[0]):
        if len(reps) >= n:
            break
        if not bool(ok[i]):
            continue
        fi = feats[i]
        if chosen_feats:
            d = torch.stack([(fi - cf).pow(2).mean().sqrt() for cf in chosen_feats])
            if float(d.min()) < dedup_eps:
                continue
        reps.append(imgs[i]); chosen_feats.append(fi)
    n_real = len(reps)
    if len(reps) < n and bank is not None:
        bf = S.feat_vectors(bank)
        fill = S.farthest_point_fill(bank, bf, chosen_feats, n - len(reps))
        reps += [bank[j] for j in fill]
    while len(reps) < n:
        reps.append(reps[len(reps) % max(1, len(reps))])
    sel = torch.stack(reps[:n]).clamp(0, 1).float()
    # LIGHT denoise: only the grainiest slots move a little
    sel = S.adaptive_denoise(sel, tv_lo=0.16, tv_hi=0.34, max_alpha=0.35, sigma=1.0)
    return sel, n_real


def main():
    c3 = torch.load(C3, weights_only=False)
    bank = S.load_bank()
    sub = {m: c3[m].float().clamp(0, 1).cpu() for m in U.MODELS}

    m6 = torch.load(U.work_path("candidates_v5", "model6", "conv_invert.pt"),
                    weights_only=False)["images"].float()
    sel6, n_real = select_faces(m6, bank, tv_cut=0.30)
    sub["model6"] = sel6.to(torch.float32).contiguous()
    print(f"model6: kept {n_real} real grainy faces (v3 kept 46), "
          f"bank-filled {128 - min(n_real, 128)}")

    sub = {m: v.to(torch.float32).contiguous() for m, v in sub.items()}
    errs = validate_submission(sub)
    for m in U.MODELS:
        u = torch.unique(sub[m].reshape(128, -1), dim=0).shape[0]
        if u < 128:
            errs.append(f"{m}: {u}/128 unique")
    if errs:
        print("VALIDATION ERRORS:", *errs, sep="\n  - "); raise SystemExit(1)

    out_dir = U.work_path("outputs_c5")
    U.ensure_dir(out_dir)
    out = os.path.join(out_dir, "submission_c5.pt")
    torch.save(sub, out)
    U.save_image_grid(sub["model6"], os.path.join(out_dir, "model6_c5_grid.png"), ncol=16)
    preview = torch.cat([sub[m][:12] for m in U.MODELS], 0)
    U.save_image_grid(preview, os.path.join(out_dir, "submission_c5_preview.png"), ncol=12)
    # confirm only model6 changed vs c3
    changed = [m for m in U.MODELS if not torch.equal(sub[m], c3[m].float().clamp(0, 1).cpu())]
    print("changed-vs-c3:", changed)
    print("saved", out, "VALID")


if __name__ == "__main__":
    main()
