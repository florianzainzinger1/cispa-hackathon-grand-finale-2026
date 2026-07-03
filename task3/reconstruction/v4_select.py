"""v4 selection.

Only the three strided CNN models (2, 3, 7) change vs v3 — they are re-selected
from the v4 conv-inversion pool (base_res tuned to ~2x feature res, which the
synthetic Hungarian-SSIM proxy showed beats both v3 and the face bank). Every
other model reuses its v3 selection verbatim (identical pipeline).

For the strided models the pool is: v4 conv-invert candidates only (the v3
selected_128 for these models was built from the WORSE base_res inversion, so
including it would re-introduce the lower-quality faces). Noise-filter + greedy
dedup to keep the distinct blurry faces, then fill any shortfall from the bank.
"""
import os
import sys
import json
import shutil
import torch

import utils as U
import v3_select as S
from postprocess import adaptive_denoise

STRIDED = ["model2", "model3", "model7"]


def load_bank():
    """Prefer the v4 bank (model6 excluded), fall back to v3, then v1."""
    for p in (U.work_path("outputs_v4", "face_bank_v4.pt"),
              U.work_path("outputs_v3", "face_bank_v3.pt"),
              U.work_path("outputs", "face_bank.pt")):
        if os.path.exists(p):
            return torch.load(p, weights_only=False)["images"].float().clamp(0, 1)
    return None


def load_v4_pool(name):
    """v4 conv-invert candidates for a strided model."""
    p = U.work_path("candidates_v4", name, "conv_invert_candidates.pt")
    if not os.path.exists(p):
        return None
    return torch.load(p, weights_only=False)["images"].float().clamp(0, 1)


def select_strided(name, n=128, tv_cut=0.22, dedup_eps=0.03, verbose=True):
    imgs = load_v4_pool(name)
    bank = load_bank()
    info = {"name": name, "pool_source": "conv_invert_v4"}
    reps, have_feats = [], []
    if imgs is not None:
        q, st = S.quality_and_stats(imgs)
        real = S.is_real(st, tv_cut=tv_cut)
        feats = S.feat_vectors(imgs)
        idx_real = torch.nonzero(real, as_tuple=False).squeeze(1)
        info["pool_size"] = int(imgs.shape[0])
        info["real_after_noise_filter"] = int(real.sum())
        if idx_real.numel() > 0:
            sub_imgs, sub_q, sub_feats = imgs[idx_real], q[idx_real], feats[idx_real]
            chosen_local, chosen_feats = S.greedy_dedup(sub_imgs, sub_q, sub_feats,
                                                        n=n, dedup_eps=dedup_eps)
            for li in chosen_local:
                reps.append(sub_imgs[li])
            have_feats = chosen_feats
    info["from_reconstruction"] = len(reps)

    n_needed = n - len(reps)
    filled = 0
    if n_needed > 0 and bank is not None:
        bank_feats = S.feat_vectors(bank)
        fill_idx = S.farthest_point_fill(bank, bank_feats, have_feats, n_needed)
        for j in fill_idx:
            reps.append(bank[j])
        filled = len(fill_idx)
    info["from_bank"] = filled
    while len(reps) < n:
        base = reps[len(reps) % max(1, len(reps))] if reps else torch.rand(3, 64, 64)
        g = torch.Generator().manual_seed(len(reps))
        reps.append((base + 0.02 * torch.randn(3, 64, 64, generator=g)).clamp(0, 1))
    info["padded"] = max(0, n - info["from_reconstruction"] - info["from_bank"])

    sel = torch.stack(reps[:n]).clamp(0, 1).float()
    # strided recons are already smooth (low-res upsample); keep denoise gentle so
    # we do not over-blur the genuine coarse structure that scores the SSIM.
    sel = adaptive_denoise(sel, tv_lo=0.14, tv_hi=0.30, max_alpha=0.4, sigma=1.0)
    out_dir = U.work_path("candidates_v4", name)
    U.ensure_dir(out_dir)
    torch.save({"images": sel, "info": info},
               os.path.join(out_dir, "selected_128.pt"))
    U.save_image_grid(sel, os.path.join(out_dir, "grid_selected.png"), ncol=16)
    if verbose:
        print(json.dumps(info), flush=True)
    return sel, info


def copy_v3(name):
    """Reuse the v3 selection unchanged for non-strided models."""
    src = U.work_path("candidates_v3", name, "selected_128.pt")
    dst_dir = U.work_path("candidates_v4", name)
    U.ensure_dir(dst_dir)
    dst = os.path.join(dst_dir, "selected_128.pt")
    if not os.path.exists(src):
        return None, {"name": name, "pool_source": "MISSING_v3"}
    blob = torch.load(src, weights_only=False)
    torch.save(blob, dst)
    info = dict(blob.get("info", {}))
    info["pool_source"] = "v3_copy"
    return blob["images"], info


# models whose v3 selection depended on the (old) bank: regenerate with v4 bank
BANK_DEPENDENT = ["model6", "model9", "model11"]


def select_pure_bank(name, n=128):
    """model9/11 (ViT): no per-sample reconstruction -> the whole v4 bank."""
    bank = load_bank()
    out_dir = U.work_path("candidates_v4", name)
    U.ensure_dir(out_dir)
    if bank is None or bank.shape[0] < n:
        # fall back to copying v3
        return copy_v3(name)
    sel = bank[:n].clamp(0, 1).float()
    info = {"name": name, "pool_source": "v4_bank", "from_reconstruction": 0,
            "from_bank": n, "padded": 0}
    torch.save({"images": sel, "info": info}, os.path.join(out_dir, "selected_128.pt"))
    U.save_image_grid(sel, os.path.join(out_dir, "grid_selected.png"), ncol=16)
    return sel, info


def select_recon_plus_bank(name, n=128, tv_cut=0.22, dedup_eps=0.03):
    """model6: keep its (few) real stride-1 reconstructions, fill from v4 bank.
    Same as v3_select.select_model but with the v4 bank."""
    imgs, src = S.load_pool(name)
    bank = load_bank()
    info = {"name": name, "pool_source": src}
    reps, have_feats = [], []
    if imgs is not None:
        q, st = S.quality_and_stats(imgs)
        real = S.is_real(st, tv_cut=tv_cut)
        feats = S.feat_vectors(imgs)
        idx_real = torch.nonzero(real, as_tuple=False).squeeze(1)
        info["real_after_noise_filter"] = int(real.sum())
        if idx_real.numel() > 0:
            sub_imgs, sub_q, sub_feats = imgs[idx_real], q[idx_real], feats[idx_real]
            chosen_local, chosen_feats = S.greedy_dedup(sub_imgs, sub_q, sub_feats,
                                                        n=n, dedup_eps=dedup_eps)
            for li in chosen_local:
                reps.append(sub_imgs[li])
            have_feats = chosen_feats
    info["from_reconstruction"] = len(reps)
    n_needed = n - len(reps)
    filled = 0
    if n_needed > 0 and bank is not None:
        bank_feats = S.feat_vectors(bank)
        fill_idx = S.farthest_point_fill(bank, bank_feats, have_feats, n_needed)
        for j in fill_idx:
            reps.append(bank[j])
        filled = len(fill_idx)
    info["from_bank"] = filled
    while len(reps) < n:
        base = reps[len(reps) % max(1, len(reps))] if reps else torch.rand(3, 64, 64)
        g = torch.Generator().manual_seed(len(reps))
        reps.append((base + 0.02 * torch.randn(3, 64, 64, generator=g)).clamp(0, 1))
    info["padded"] = max(0, n - info["from_reconstruction"] - info["from_bank"])
    sel = torch.stack(reps[:n]).clamp(0, 1).float()
    sel = adaptive_denoise(sel, tv_lo=0.12, tv_hi=0.28, max_alpha=0.6, sigma=1.0)
    out_dir = U.work_path("candidates_v4", name)
    U.ensure_dir(out_dir)
    torch.save({"images": sel, "info": info}, os.path.join(out_dir, "selected_128.pt"))
    U.save_image_grid(sel, os.path.join(out_dir, "grid_selected.png"), ncol=16)
    return sel, info


def main():
    names = sys.argv[1:] or U.MODELS
    allinfo = {}
    for name in names:
        try:
            if name in STRIDED:
                _, info = select_strided(name)
            elif name in ("model9", "model11"):
                _, info = select_pure_bank(name)
            elif name == "model6":
                _, info = select_recon_plus_bank(name)
            else:
                _, info = copy_v3(name)
            allinfo[name] = info
            print(f"{name}: {info.get('pool_source')} "
                  f"recon={info.get('from_reconstruction','-')} bank={info.get('from_bank','-')}")
        except Exception as e:
            import traceback
            print(f"{name}: ERROR {e}")
            traceback.print_exc()
    U.save_json(allinfo, U.work_path("candidates_v4", "_select_summary.json"))
    print("wrote", U.work_path("candidates_v4", "_select_summary.json"))


if __name__ == "__main__":
    main()
