"""Agent R / STEP 2 — the DECISIVE verbatim-membership test.

For each MLP model (net.0 input == pixel image) and each crop gallery:
  (A) recon-guided NN: SSIM of each analytic recon query vs gallery (orig+flip),
      MSE-prefiltered then SSIM-refined. Read the best-SSIM histogram.
  (B) recon-free subspace scan: residual of every gallery image onto the top-128
      right-singular subspace of net.0 weight-grad. True members ~0, look-alikes high.

Verdict logic (RESEARCH_FINDINGS.md §2):
  * spike best-SSIM > 0.95 at one crop AND matched-original subspace-residual ~0
      -> VERBATIM, no aug. Accept overrides.
  * plateau 0.5-0.8, flips win, no >0.95 spike -> right dataset but AUG ON -> verbatim
      dead (identity-prior only).
  * no crop beats the within-set nearest-different-face baseline -> synthetic -> NO.

Writes reports/retr/membership.json and compare grids. Read-only w.r.t. everything
except reports/retr and outputs_retr.
"""
import os, sys, json, time
import numpy as np
import torch

TEAM = "/p/scratch/training2625/zainzinger1/1337"
WORK = os.environ.get("FL_WORK_ROOT", f"{TEAM}/recon_work")
DATA = os.environ.get("FL_DATA_ROOT", f"{TEAM}/FL_Data_Reconstruction")
sys.path.insert(0, os.path.join(WORK, "code"))
from utils import save_image_grid
from retr_lib import (load_grad_firstlayer, topk_right_subspace,
                      subspace_residual, augment_dc)
from skimage.metrics import structural_similarity as ssim

MODELS = os.environ.get("RETR_MODELS", "model1,model4,model5,model8").split(",")
CROPS = os.environ.get("RETR_CROPS", "cc148,cc128,cc178,r64").split(",")
GALDIR = os.path.join(WORK, "gallery")
REP = os.path.join(WORK, "reports", "retr")
os.makedirs(REP, exist_ok=True)
os.makedirs(os.path.join(WORK, "outputs_retr"), exist_ok=True)
SSIM_ACCEPT = float(os.environ.get("RETR_SSIM", "0.95"))
PREFILTER_K = int(os.environ.get("RETR_PREK", "40"))


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_query(m):
    """Cleanest MLP recon for model m. Prefer agent2 (outputs_trap), else v3."""
    cands = [
        os.path.join(WORK, "outputs_trap", f"trap_{m}.pt"),
        os.path.join(WORK, "candidates_v3", m, "refined.pt"),
        os.path.join(WORK, "candidates_v3", m, "selected_128.pt"),
        os.path.join(WORK, "candidates", m, "selected_128.pt"),
    ]
    for p in cands:
        if os.path.exists(p):
            obj = torch.load(p, weights_only=False, map_location="cpu")
            x = obj["images"] if isinstance(obj, dict) and "images" in obj else obj
            if torch.is_tensor(x) and x.ndim == 4:
                return x.float().clamp(0, 1), p
    return None, None


def perimg_standardize(X):
    """X (N,D) -> zero-mean unit-std per row (affine-robust for MSE prefilter)."""
    mu = X.mean(dim=1, keepdim=True)
    sd = X.std(dim=1, keepdim=True) + 1e-6
    return (X - mu) / sd


def ssim_pair(a, b):
    a = a.permute(1, 2, 0).numpy()
    b = b.permute(1, 2, 0).numpy()
    return ssim(a, b, channel_axis=2, data_range=1.0)


def nn_ssim(Q, G, prek=PREFILTER_K):
    """For each query in Q (nq,3,64,64), MSE-prefilter to top-prek gallery images
    (G, ng,3,64,64), SSIM-refine. Returns best_ssim (nq,), best_idx (nq,)."""
    nq = Q.shape[0]
    Qf = perimg_standardize(Q.reshape(nq, -1))
    Gf = perimg_standardize(G.reshape(G.shape[0], -1))
    best_s = np.full(nq, -1.0)
    best_i = np.full(nq, -1, dtype=np.int64)
    # chunked MSE prefilter
    for qi in range(nq):
        d = ((Gf - Qf[qi:qi + 1]) ** 2).mean(dim=1)
        cand = torch.argsort(d)[:prek].tolist()
        qimg = Q[qi]
        for gi in cand:
            s = ssim_pair(qimg, G[gi])
            if s > best_s[qi]:
                best_s[qi] = s
                best_i[qi] = gi
    return best_s, best_i


def main():
    report = {"models": {}, "crops": CROPS, "ssim_accept": SSIM_ACCEPT}
    galleries = {}
    for c in CROPS:
        p = os.path.join(GALDIR, f"celeba_top200_{c}.pt")
        if os.path.exists(p):
            g = torch.load(p, weights_only=False, map_location="cpu")
            galleries[c] = g["images"].float() / 255.0
            log(f"gallery {c}: {tuple(galleries[c].shape)}")
        else:
            log(f"gallery {c}: MISSING ({p})")
    if not galleries:
        log("NO galleries built yet; aborting"); return

    for m in MODELS:
        Q, qpath = load_query(m)
        if Q is None:
            log(f"{m}: no query recon found; skip"); continue
        gW, gb, key = load_grad_firstlayer(DATA, m)
        if key != "net.0.weight":
            log(f"{m}: first-layer key={key} (not MLP pixel space); skip subspace");
        V, S = topk_right_subspace(gW, 128)
        Vaug = augment_dc(V)
        # baselines on the subspace: random [0,1] faces
        r_rand = subspace_residual(torch.rand(256, 3, 64, 64), V).mean().item()
        r_rand_a = subspace_residual(torch.rand(256, 3, 64, 64), Vaug).mean().item()
        # within-query nearest-different-face SSIM baseline (a few samples)
        wb = []
        for qi in range(0, min(24, Q.shape[0])):
            best = -1
            for qj in range(Q.shape[0]):
                if qj == qi:
                    continue
                best = max(best, ssim_pair(Q[qi], Q[qj]))
            wb.append(best)
        within_base = float(np.mean(wb)) if wb else 0.0

        mrep = {"query": qpath, "within_diff_ssim_base": round(within_base, 4),
                "rand_resid": round(r_rand, 4), "rand_resid_aug": round(r_rand_a, 4),
                "crops": {}}
        log(f"\n=== {m}  query={os.path.relpath(qpath, WORK)}  "
            f"within-base SSIM={within_base:.3f} rand-resid={r_rand:.3f} ===")
        best_crop, best_spike = None, -1
        for c, G in galleries.items():
            bs, bi = nn_ssim(Q, G)
            matched = G[torch.from_numpy(bi)]
            res = subspace_residual(matched, V).numpy()
            res_a = subspace_residual(matched, Vaug).numpy()
            gmeta = torch.load(os.path.join(GALDIR, f"celeba_top200_{c}.pt"),
                               weights_only=False, map_location="cpu")
            flips = gmeta["is_flip"].numpy()[bi]
            n95 = int((bs > 0.95).sum()); n90 = int((bs > 0.90).sum())
            n80 = int((bs > 0.80).sum())
            # accepted = SSIM>accept AND low residual (< 0.5*rand, robust via aug)
            acc = (bs > SSIM_ACCEPT) & (res_a < 0.5 * r_rand_a)
            cr = {"ssim_max": round(float(bs.max()), 4),
                  "ssim_mean": round(float(bs.mean()), 4),
                  "ssim_p90": round(float(np.percentile(bs, 90)), 4),
                  "n_ssim>0.95": n95, "n>0.90": n90, "n>0.80": n80,
                  "flip_frac": round(float(flips.mean()), 3),
                  "matched_resid_mean": round(float(res.mean()), 4),
                  "matched_resid_aug_mean": round(float(res_a.mean()), 4),
                  "matched_resid_min": round(float(res.min()), 4),
                  "n_accepted(ssim&resid)": int(acc.sum())}
            mrep["crops"][c] = cr
            log(f"  {c}: SSIM max{cr['ssim_max']:.3f} mean{cr['ssim_mean']:.3f} "
                f">0.95={n95} >0.90={n90} >0.80={n80} flip%={cr['flip_frac']:.2f} "
                f"| resid_aug{cr['matched_resid_aug_mean']:.3f}(rand{r_rand_a:.3f}) "
                f"accepted={cr['n_accepted(ssim&resid)']}")
            spike = n95 + 0.2 * n90
            if spike > best_spike:
                best_spike, best_crop = spike, c
            # save a compare grid: query vs matched (top 32 by ssim)
            order = np.argsort(-bs)[:32]
            inter = torch.stack([torch.stack([Q[i], matched[i]]) for i in order])
            inter = inter.reshape(-1, 3, 64, 64)
            save_image_grid(inter, os.path.join(REP, f"cmp_{m}_{c}.png"), ncol=8)
        mrep["best_crop"] = best_crop
        report["models"][m] = mrep

    with open(os.path.join(REP, "membership.json"), "w") as f:
        json.dump(report, f, indent=2)
    log("\nWROTE reports/retr/membership.json")

    # ---- verdict summary ----
    print("\n===== VERDICT SUMMARY =====")
    any_verbatim = False
    for m, mr in report["models"].items():
        best = mr.get("best_crop")
        if best is None:
            continue
        cr = mr["crops"][best]
        verdict = ("VERBATIM" if cr["n_ssim>0.95"] >= 3 and cr["matched_resid_aug_mean"] < 0.5 * mr["rand_resid_aug"]
                   else ("AUG-ON/prior" if cr["ssim_max"] > mr["within_diff_ssim_base"] + 0.1
                         else "no-signal"))
        any_verbatim = any_verbatim or verdict == "VERBATIM"
        print(f"  {m}: best={best} maxSSIM={cr['ssim_max']:.3f} "
              f">0.95={cr['n_ssim>0.95']} accepted={cr['n_accepted(ssim&resid)']} "
              f"resid_aug={cr['matched_resid_aug_mean']:.3f}/rand{mr['rand_resid_aug']:.3f} "
              f"within-base={mr['within_diff_ssim_base']:.3f} -> {verdict}")
    print(f"\nANY VERBATIM SPIKE: {any_verbatim}")
    print("__DONE__")


if __name__ == "__main__":
    main()
