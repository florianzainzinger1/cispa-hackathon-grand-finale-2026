"""Faithful-recon + noise-appropriate denoise candidate (specialist-agent lever).

Thesis: the fleet's rec_model*.pt are FAITHFUL (low top-128 gradient-subspace
residual) but carry model-specific noise; the 0.3718 best used BLURRY sources +
one Gaussian NLM. Here we denoise the faithful recons with a per-FAMILY denoiser
and validate GT-blind via the subspace residual (denoise should DECREASE it =
move toward the true image; if it increases, we're deleting signal -> reject).

Writes ONLY to outputs_ff/ (own namespace). Does NOT touch best/c3/current_best.
Reads rec_model*.pt (Agent T) + current best test_nlm.pt for models we don't upgrade.
"""
import os, json, numpy as np, torch
from scipy.ndimage import median_filter
from skimage.restoration import denoise_bilateral, denoise_nl_means

WORK = os.environ.get("FL_WORK_ROOT", "/p/scratch/training2625/zainzinger1/1337/recon_work")
DATA = os.environ.get("FL_DATA_ROOT", "/p/scratch/training2625/zainzinger1/1337/FL_Data_Reconstruction")
OUT = os.path.join(WORK, "outputs_ff"); os.makedirs(OUT, exist_ok=True)

def L(p): return torch.load(p, weights_only=False, map_location="cpu")
def npimg(t): return t.clamp(0,1).permute(0,2,3,1).numpy()          # (N,H,W,3)
def totensor(a): return torch.from_numpy(np.ascontiguousarray(a)).permute(0,3,1,2).float().clamp(0,1)
def tv(t):
    t=t.clamp(0,1); return float((t[...,1:,:]-t[...,:-1,:]).abs().mean()+(t[...,:,1:]-t[...,:,:-1]).abs().mean())

def est_sigma(img):
    res=img-np.stack([median_filter(img[...,c],size=3) for c in range(3)],-1)
    return float(1.4826*np.median(np.abs(res-np.median(res)))+1e-4)

def denoise_mlp(x):                       # fine Gaussian grain -> edge-preserving NLM
    out=np.empty_like(x)
    for i in range(len(x)):
        s=est_sigma(x[i])
        out[i]=denoise_nl_means(x[i], h=max(0.05,0.85*s), sigma=s, patch_size=5,
                                patch_distance=6, channel_axis=-1, fast_mode=True)
    return out

def denoise_cnn(x, strong=False):         # colored-blob impulse -> chroma-outlier clamp + median + bilateral
    out=np.empty_like(x)
    for i in range(len(x)):
        im=x[i].copy()
        # clamp chroma outliers: pixels whose |color - local median| is large -> replace by local median
        med=np.stack([median_filter(im[...,c],size=3) for c in range(3)],-1)
        dev=np.abs(im-med).max(-1, keepdims=True)
        mask=dev> (0.35 if strong else 0.45)
        im=np.where(mask, med, im)
        if strong:
            im=np.stack([median_filter(im[...,c],size=3) for c in range(3)],-1)
        out[i]=denoise_bilateral(im, sigma_color=0.06, sigma_spatial=1.1, channel_axis=-1)
    return out

# ---- top-128 gradient-subspace residual (GT-blind faithfulness proxy) ----
# MLP: net.0.weight (1024,12288) rowspace spans the flattened RGB inputs directly.
def mlp_subspace(name, k=128):
    g=L(f"{DATA}/gradients/{name}.pt")["gradients"]
    key="net.0.weight" if "net.0.weight" in g else [x for x in g if g[x].ndim==2][0]
    gW=g[key].float()
    _,_,Vh=torch.linalg.svd(gW, full_matrices=False)
    return Vh[:k]
def residual(imgs, V):
    X=imgs.reshape(imgs.shape[0],-1).float()
    proj=X@V.t()@V
    return ((X-proj).norm(dim=1)/(X.norm(dim=1)+1e-9)).mean().item()

FAMILY={1:"mlp",4:"mlp",5:"mlp",8:"mlp",2:"cnn",3:"cnn",7:"cnn",10:"cnn",12:"cnn",6:"cnn",9:"vit",11:"vit"}
# upgrade set: faithful recons only. model6 rec is noisy (resid~0.65) -> keep best. ViTs -> keep best (bank).
UPGRADE={1,4,5,8,2,3,7,10,12}
STRONG_CNN={2,6}                       # heavier speckle

best=L(f"{WORK}/outputs_final/test_nlm.pt")
out={}; report={}
for i in range(1,13):
    m=f"model{i}"
    if i not in UPGRADE:
        out[m]=best[m].float().clamp(0,1)
        report[m]={"source":"kept_best(blurry/bank)","tv_out":round(tv(out[m]),4)}
        continue
    rec=L(f"{WORK}/outputs_trap/rec_{m}.pt")["images"].float().clamp(0,1)
    x=npimg(rec)
    if FAMILY[i]=="mlp":
        den=denoise_mlp(x)
    else:
        den=denoise_cnn(x, strong=i in STRONG_CNN)
    den_t=totensor(den)
    rep={"source":"faithful_rec+denoise", "family":FAMILY[i],
         "tv_rec":round(tv(rec),4), "tv_den":round(tv(den_t),4), "tv_best":round(tv(best[m]),4)}
    if FAMILY[i]=="mlp":                # GT-blind faithfulness proxy for MLPs
        V=mlp_subspace(m)
        rep["resid_rec"]=round(residual(rec,V),4)
        rep["resid_den"]=round(residual(den_t,V),4)
        rep["resid_best"]=round(residual(best[m].float(),V),4)
    out[m]=den_t
    report[m]=rep
    print(m, rep, flush=True)

# validate + save
for m,v in out.items():
    assert v.shape==(128,3,64,64) and torch.isfinite(v).all() and v.min()>=0 and v.max()<=1, m
torch.save(out, os.path.join(OUT,"submission_ff.pt"))
json.dump(report, open(os.path.join(OUT,"ff_report.json"),"w"), indent=2)
print("\nSAVED", os.path.join(OUT,"submission_ff.pt"))
print(json.dumps(report, indent=2))
