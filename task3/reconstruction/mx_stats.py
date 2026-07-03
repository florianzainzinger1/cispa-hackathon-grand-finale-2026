import torch, os

WORK = os.environ.get("FL_WORK_ROOT", "/p/scratch/training2625/zainzinger1/1337/recon_work")

def stats(name, t):
    # t: (N,3,H,W) in [0,1]
    t = t.float()
    m = t.mean(dim=(0,2,3))
    s = t.std(dim=(0,2,3))
    gm = t.mean().item()
    lo, hi = t.min().item(), t.max().item()
    # per-image brightness spread
    ib = t.mean(dim=(1,2,3))
    print(f"{name:28s} shape={tuple(t.shape)} gmean={gm:.4f} rng[{lo:.3f},{hi:.3f}]")
    print(f"{'':28s} chan_mean=[{m[0]:.4f},{m[1]:.4f},{m[2]:.4f}] chan_std=[{s[0]:.4f},{s[1]:.4f},{s[2]:.4f}]")
    print(f"{'':28s} per_img_bright min={ib.min():.3f} med={ib.median():.3f} max={ib.max():.3f}")

# CelebA gallery reference
g = torch.load(f"{WORK}/gallery/celeba_top200_cc178.pt", map_location="cpu")
if isinstance(g, dict):
    print("gallery keys:", list(g.keys())[:8])
    # find the image tensor
    for k,v in g.items():
        if torch.is_tensor(v) and v.dim()==4:
            g = v; print("using gallery key", k); break
if g.max() > 1.5:
    print(">> gallery in [0,255], dividing by 255")
    g = g/255.0
stats("CELEBA_GALLERY(ref)", g)
print()

# vithi45 submission
sub = torch.load(f"{WORK}/outputs_final/test_vithi45.pt", map_location="cpu")
print("vithi45 keys:", sorted(sub.keys()))
for mk in ["model10","model12","model2","model3","model7"]:
    stats(f"vithi45_{mk}", sub[mk])
print()

# raw trap recons
for rk in ["rec_model12","rec_model10"]:
    d = torch.load(f"{WORK}/outputs_trap/{rk}.pt", map_location="cpu")
    if isinstance(d, dict):
        print(f"{rk} keys:", list(d.keys()))
        img = d.get("images", None)
        if img is None:
            for k,v in d.items():
                if torch.is_tensor(v) and v.dim()==4: img=v; break
    else:
        img = d
    stats(f"trap_{rk}", img)
