"""Inspect outputs_retr/ground_truth.pt (per-model dict). Descend into each
model's sub-dict, stats the image tensor, run gradient-subspace consistency and
a gradient-replay cosine to decide if it's GENUINE retrieved ground truth.
Read-only."""
import os, sys, json, torch
import numpy as np

TEAM = "/p/scratch/training2625/zainzinger1/1337"
WORK = os.environ.get("FL_WORK_ROOT", f"{TEAM}/recon_work")
DATA = os.environ.get("FL_DATA_ROOT", f"{TEAM}/FL_Data_Reconstruction")
sys.path.insert(0, os.path.join(WORK, "code"))
from utils import save_image_grid, total_variation
from retr_lib import topk_right_subspace, subspace_residual

gt = torch.load(os.path.join(WORK, "outputs_retr", "ground_truth.pt"),
                weights_only=False, map_location="cpu")
print("top keys:", list(gt))

def find_images(d):
    """Return the (128,3,64,64)-ish image tensor inside a per-model value."""
    if torch.is_tensor(d):
        return d
    if isinstance(d, dict):
        for k in ("images", "imgs", "retrieved", "tensor", "x", "data", "gallery"):
            if k in d and torch.is_tensor(d[k]):
                return d[k]
        # fallback: first 4D float tensor
        for k, v in d.items():
            if torch.is_tensor(v) and v.ndim == 4:
                return v
    return None

# show substructure of first model
m0 = list(gt)[0]
print(f"\n{m0} sub-structure:")
if isinstance(gt[m0], dict):
    for k, v in gt[m0].items():
        desc = tuple(v.shape) if torch.is_tensor(v) else (v if not isinstance(v, (list, dict)) else f"{type(v).__name__}[{len(v)}]")
        print(f"    {k:14s} {type(v).__name__:10s} {desc}")

def gw_subspace(name, k=128):
    g = torch.load(os.path.join(DATA, "gradients", f"{name}.pt"),
                   weights_only=False, map_location="cpu")["gradients"]
    key = "net.0.weight" if "net.0.weight" in g else list(g)[0]
    gW = g[key].float()
    V, S = topk_right_subspace(gW, k)
    return V, key

def residual(imgs, V):
    return subspace_residual(imgs, V)

os.makedirs(os.path.join(WORK, "outputs_retr", "gt_grids"), exist_ok=True)
print("\nPER-MODEL:")
MLPS = {"model1", "model4", "model5", "model8"}
for m in list(gt):
    imgs = find_images(gt[m])
    if imgs is None:
        print(f"  {m}: NO image tensor found; keys={list(gt[m]) if isinstance(gt[m],dict) else type(gt[m])}")
        continue
    imgs = imgs.float()
    tv = total_variation(imgs.clamp(0, 1)).mean().item()
    line = (f"  {m}: {tuple(imgs.shape)} range[{imgs.min():.3f},{imgs.max():.3f}] "
            f"mean{imgs.mean():.3f} tv{tv:.3f}")
    if m in MLPS:
        V, key = gw_subspace(m)
        r_gt = residual(imgs, V)
        r_rand = residual(torch.rand(128, 3, 64, 64), V)
        line += f"  | SUBSPACE-RES gt {r_gt.mean():.4f}±{r_gt.std():.3f} vs rand {r_rand.mean():.3f}"
    print(line)
    save_image_grid(imgs[:64].clamp(0, 1),
                    os.path.join(WORK, "outputs_retr", "gt_grids", f"gt_{m}.png"))

# also dump any scalar metadata per model (crop, dataset, residual, score...)
print("\nMETADATA per model:")
for m in list(gt):
    if isinstance(gt[m], dict):
        meta = {k: v for k, v in gt[m].items()
                if not torch.is_tensor(v) and not isinstance(v, (list, dict))}
        listmeta = {k: (len(v) if hasattr(v, "__len__") else v)
                    for k, v in gt[m].items() if isinstance(v, list)}
        print(f"  {m}: {meta} {('lists='+str(listmeta)) if listmeta else ''}")
print("\ngrids -> outputs_retr/gt_grids/  DONE.")
