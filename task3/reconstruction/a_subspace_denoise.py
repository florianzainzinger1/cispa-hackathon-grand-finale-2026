"""runs/a* — Direction B: faithful subspace-projection denoising for MLP models.

Theory: the batch gradient gW = A X (rank<=128); its top right-singular vectors
V_k span the row space = the span of the 128 private images. A noisy candidate
c = x_true + noise; the DP-style noise has large energy ORTHOGONAL to span(V_k),
so projecting c onto span(V_k) removes that noise without inventing identity
(faithful: it can only move c within the true image span). Blend alpha, clamp.

Sweep k, alpha; measure GT-free faithfulness (TV/grain down, distinct-count kept,
subspace residual, gradient self-consistency). Writes runs/<RUN>/ only. Read-only
w.r.t. all existing submissions/candidates.

Env: A_RUN (default a2), A_MODELS (default model8), A_SRC (base submission),
A_KS, A_ALPHAS.
"""
import os, sys, json, itertools
import numpy as np, torch

TEAM = "/p/scratch/training2625/zainzinger1/1337"
WORK = os.environ.get("FL_WORK_ROOT", f"{TEAM}/recon_work")
DATA = os.environ.get("FL_DATA_ROOT", f"{TEAM}/FL_Data_Reconstruction")
sys.path.insert(0, os.path.join(WORK, "code"))
import utils as U
from retr_lib import topk_right_subspace
from rec_lib import total_variation

RUN = os.environ.get("A_RUN", "a2")
MODELS = os.environ.get("A_MODELS", "model8").split(",")
SRC = os.environ.get("A_SRC", "outputs_v3/submission_v3.pt")
KS = [int(x) for x in os.environ.get("A_KS", "96,128,160,192,256").split(",")]
ALPHAS = [float(x) for x in os.environ.get("A_ALPHAS", "0.5,0.75,1.0").split(",")]
RUNDIR = os.path.join(WORK, "runs", RUN)
for sub in ("candidates", "grids", "reports", "logs"):
    os.makedirs(os.path.join(RUNDIR, sub), exist_ok=True)


def log(m):
    print(m, flush=True)


def distinct_count(x, thr=0.04):
    g = torch.nn.functional.interpolate(x.mean(1, keepdim=True), size=(16, 16),
                                        mode="bilinear", align_corners=False)
    g = g.reshape(g.shape[0], -1)
    kept = []
    for i in range(g.shape[0]):
        if all(((g[i]-g[j])**2).mean().sqrt() > thr for j in kept):
            kept.append(i)
    return len(kept)


def project(imgs, Vk):
    X = imgs.reshape(imgs.shape[0], -1).float()
    return ((X @ Vk.t()) @ Vk).reshape_as(imgs)


base = torch.load(os.path.join(WORK, SRC), weights_only=False, map_location="cpu")
report = {}
for m in MODELS:
    g = U.load_grad(m)["gradients"]
    key = "net.0.weight" if "net.0.weight" in g else "fc1.weight"
    if key != "net.0.weight":
        log(f"{m}: not MLP image-space (key={key}); skip (use feature-space script)")
        continue
    gW = g[key].float()
    orig = base[m]
    orig = (orig.get("images", orig) if isinstance(orig, dict) else orig).float().clamp(0, 1)
    tv0 = total_variation(orig).mean().item()
    d0 = distinct_count(orig)
    log(f"\n=== {m}: base TV={tv0:.4f} distinct={d0} ===")
    mrep = {"base_tv": round(tv0, 4), "base_distinct": d0, "sweep": {}}
    # precompute the biggest-k subspace once, slice for smaller k
    Vbig, S = topk_right_subspace(gW, max(KS))
    best = None
    for k, a in itertools.product(KS, ALPHAS):
        Vk = Vbig[:k]
        proj = project(orig, Vk)
        new = (a * proj + (1 - a) * orig).clamp(0, 1)
        tv = total_variation(new).mean().item()
        dist = distinct_count(new)
        # faithfulness: TV drop (grain removed) while keeping distinctness; also
        # how far we moved (should be modest — denoise, not reinvent)
        move = (new - orig).reshape(128, -1).norm(dim=1).mean().item()
        tag = f"k{k}_a{a}"
        mrep["sweep"][tag] = {"tv": round(tv, 4), "distinct": dist,
                              "mean_move_L2": round(move, 3)}
        # heuristic score: prefer TV reduction, keep distinct>=0.9*base, small move
        ok = dist >= 0.9 * d0
        score = (tv0 - tv) - 0.5 * max(0, move - 3.0) - (0 if ok else 10)
        if best is None or score > best[0]:
            best = (score, tag, new.clone(), k, a)
        log(f"  {tag}: TV {tv:.4f} (base {tv0:.4f}) distinct={dist} move={move:.2f}")
    # save the best candidate + a comparison grid
    score, tag, newimgs, k, a = best
    torch.save({"images": newimgs, "source": f"subspace_denoise:{SRC}:{tag}",
                "model": m, "k": k, "alpha": a},
               os.path.join(RUNDIR, "candidates", f"{m}_denoise_{tag}.pt"))
    inter = torch.stack([torch.stack([orig[i], newimgs[i]]) for i in range(32)]).reshape(-1, 3, 64, 64)
    U.save_image_grid(inter, os.path.join(RUNDIR, "grids", f"{m}_orig_vs_denoise_{tag}.png"), ncol=8)
    U.save_image_grid(newimgs[:64], os.path.join(RUNDIR, "grids", f"{m}_denoise_{tag}.png"))
    mrep["best"] = {"tag": tag, "k": k, "alpha": a,
                    "candidate": f"runs/{RUN}/candidates/{m}_denoise_{tag}.pt"}
    report[m] = mrep
    log(f"{m}: BEST {tag} -> saved candidate + grids")

json.dump(report, open(os.path.join(RUNDIR, "reports", "denoise_sweep.json"), "w"), indent=2)
log(f"\nwrote runs/{RUN}/reports/denoise_sweep.json")
print("__DONE__")
