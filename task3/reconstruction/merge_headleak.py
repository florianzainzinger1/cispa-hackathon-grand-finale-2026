"""ViT HEAD-gradient CLS-embedding leak -> learned decoder (agent3, NEW lever, not in ledger).

The team wrote off ViT 9/11 because the PATCH-EMBED (first layer) leaks nothing (APRIL:
batch-mean only). But the CLASSIFIER-HEAD gradient leaks per-sample PENULTIMATE embeddings:
for a class c with iLRG count==1 (singleton), the dominant term of the CE head gradient is the
one sample n* of that class, so
    emb_c = gW_head[c] / gb_head[c] ~= f_{n*}   (exact penultimate CLS/pooled feature, scale-correct)
ViT 9/11 gradients are CLEAN (E128=1.0, verified) so these embeddings are accurate. We DECODE
emb_c -> face with a decoder D trained OFFLINE on the public CelebA gallery through the KNOWN
frozen target ViT: x -> f=ViT_penult(x); train D(f)->x with blurred-L1 + SSIM (low-freq only,
no grain). Fill the ~80 singleton ViT slots with D(emb_c); keep mean-face bank for the rest.

GT-FREE validation: the decoder's HELD-OUT reconstruction SSIM on public faces proves whether the
embedding is decodable BEFORE any leaderboard submit. Non-destructive: writes bank_headleak_*.pt +
a spliced candidate; never overwrites the champion. Heavy (decoder training) -> GPU sbatch.
"""
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils as U
import vit as VIT
import proxy_labels as PL
from build_submission import validate_submission

DEV = os.environ.get("FL_DEVICE", "cpu")
if DEV == "cuda" and not torch.cuda.is_available():
    DEV = "cpu"


# --------------------------- SSIM / blur (differentiable) ---------------------------
def _gauss_kernel(ch, win=11, sigma=1.5, device="cpu"):
    c = torch.arange(win, device=device) - win // 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    k = (g[:, None] * g[None, :]).view(1, 1, win, win).repeat(ch, 1, 1, 1)
    return k


def gblur(x, k):
    return F.conv2d(x, k, padding=k.shape[-1] // 2, groups=x.shape[1])


def ssim(a, b, k):
    mu_a, mu_b = gblur(a, k), gblur(b, k)
    va = gblur(a * a, k) - mu_a ** 2
    vb = gblur(b * b, k) - mu_b ** 2
    vab = gblur(a * b, k) - mu_a * mu_b
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_a * mu_b + C1) * (2 * vab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2))
    return s.mean()


# --------------------------- decoder ---------------------------
class Decoder(nn.Module):
    def __init__(self, d, ch=160):
        super().__init__()
        self.ch = ch
        self.fc = nn.Linear(d, ch * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(ch, ch, 4, 2, 1), nn.GELU(),           # 8
            nn.ConvTranspose2d(ch, ch // 2, 4, 2, 1), nn.GELU(),       # 16
            nn.ConvTranspose2d(ch // 2, ch // 4, 4, 2, 1), nn.GELU(),  # 32
            nn.ConvTranspose2d(ch // 4, ch // 8, 4, 2, 1), nn.GELU(),  # 64
            nn.Conv2d(ch // 8, 3, 3, 1, 1), nn.Sigmoid())

    def forward(self, f):
        return self.net(self.fc(f).view(-1, self.ch, 4, 4))


def feature_extractor(model):
    store = {}
    h = model.head.register_forward_pre_hook(lambda m, inp: store.__setitem__("f", inp[0].detach()))
    def extract(x):
        with torch.no_grad():
            model(x)
        return store["f"]
    return extract, h


def load_gallery():
    for name in ["celeba_top200_cc178.pt", "celeba_top200_cc148.pt"]:
        fp = U.work_path("gallery", name)
        if os.path.exists(fp):
            g = torch.load(fp, weights_only=False)
            t = (g["images"] if isinstance(g, dict) and "images" in g else g).float()
            if t.max() > 1.5:
                t = t / 255.0
            if t.shape[-1] != 64:
                t = F.interpolate(t, size=(64, 64), mode="bilinear", align_corners=False)
            return t.clamp(0, 1)
    return None


def eigenbasis(imgs, k=120):
    X = imgs.reshape(imgs.shape[0], -1).float()
    mean = X.mean(0, keepdim=True)
    q = min(k + 16, X.shape[0] - 1, X.shape[1])
    _, _, V = torch.pca_lowrank(X, q=q, center=True, niter=6)
    return mean, V[:, :k].t().contiguous()


def face_residual(imgs, mean, E):
    Xc = imgs.reshape(imgs.shape[0], -1) - mean
    recon = (Xc @ E.t()) @ E
    return (Xc - recon).norm(dim=1) / (Xc.norm(dim=1) + 1e-9)


def singleton_embeddings(name):
    """emb_c = gW_head[c]/gb_head[c] for iLRG count==1 classes -> {c: (d,) embedding}."""
    hw = PL.head_wb_and_grad(name)
    W, b, gW, gb = hw
    info = PL.recover(name)
    counts = torch.tensor(info["counts"])
    E = torch.tensor(PL.class_embeddings(gW, gb)).float()   # (C, d)
    singles = torch.nonzero(counts == 1, as_tuple=False).squeeze(1).tolist()
    return {c: E[c] for c in singles}, info


def train_decoder(name, model, extract, gallery, d, device, epochs=40, bs=256, log=None):
    n = gallery.shape[0]
    perm = torch.randperm(n)
    nval = min(1024, n // 10)
    val_idx, tr_idx = perm[:nval], perm[nval:]
    # precompute features for the whole gallery (frozen ViT)
    feats = []
    with torch.no_grad():
        for i in range(0, n, 512):
            feats.append(extract(gallery[i:i + 512].to(device)).cpu())
    F_all = torch.cat(feats, 0)                              # (n, d)
    D = Decoder(d).to(device)
    opt = torch.optim.Adam(D.parameters(), lr=1e-3)
    k = _gauss_kernel(3, device=device)
    kb = _gauss_kernel(3, win=7, sigma=1.2, device=device)
    best_ssim, best_state = -1.0, None
    for ep in range(epochs):
        D.train()
        idx = tr_idx[torch.randperm(tr_idx.numel())]
        for i in range(0, idx.numel(), bs):
            b_i = idx[i:i + bs]
            f = F_all[b_i].to(device); x = gallery[b_i].to(device)
            yhat = D(f)
            loss = F.l1_loss(gblur(yhat, kb), gblur(x, kb)) + (1 - ssim(yhat, x, k))
            opt.zero_grad(); loss.backward(); opt.step()
        D.eval()
        with torch.no_grad():
            vf = F_all[val_idx].to(device); vx = gallery[val_idx].to(device)
            vs = ssim(D(vf), vx, k).item()
        if vs > best_ssim:                       # early-stop: keep best-val decoder
            best_ssim = vs
            best_state = {kk: v.detach().clone() for kk, v in D.state_dict().items()}
        if ep % 8 == 0 or ep == epochs - 1:
            print(f"[{name}] ep{ep:3d} val_ssim={vs:.4f} (best={best_ssim:.4f})", flush=True)
    if best_state is not None:
        D.load_state_dict(best_state)
    return D, best_ssim


def run_model(name, gallery, base, out_dir, mean, Eig):
    state = U.load_state(name)
    model, missing, unexpected = VIT.build_vit(name, state)
    model = model.to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    extract, hook = feature_extractor(model)
    d = model.head.in_features
    embs, info = singleton_embeddings(name)
    n_single = len(embs)
    print(f"\n=== {name}: d={d} singletons={n_single} (iLRG n_distinct={info['n_distinct']}) ===", flush=True)
    D, val_ssim = train_decoder(name, model, extract, gallery, d, DEV)
    with torch.no_grad():
        if n_single > 0:
            Ecls = torch.stack([embs[c] for c in sorted(embs)]).to(DEV)
            decoded = D(Ecls).cpu().clamp(0, 1)
        else:
            decoded = torch.zeros(0, 3, 64, 64)
    hook.remove()
    slot = base[name].float().clamp(0, 1).clone()          # start from current bank
    n_bad = 0
    if n_single > 0:
        # drop off-manifold / colour-corrupted decodes (contaminated singletons)
        res = face_residual(decoded, mean, Eig)
        ch = decoded.reshape(n_single, 3, -1).mean(2)
        imbalance = ch.max(1).values / (ch.min(1).values + 1e-3)
        thr = float(res.median() + 3 * (res - res.median()).abs().median() * 1.4826)
        bad = (res > thr) | (imbalance > 2.6)
        good = decoded[~bad]
        n_bad = int(bad.sum())
        m = min(good.shape[0], 128)
        if m > 0:
            slot[:m] = good[:m]                              # remaining slots keep the base bank
    else:
        m = 0
    U.save_image_grid((good[:64] if n_single and m > 0 else slot[:64]),
                      os.path.join(out_dir, f"headleak_{name}_clean.png"))
    return slot, {"name": name, "d": d, "n_singletons": n_single,
                  "decoder_val_ssim": round(float(val_ssim), 4),
                  "n_corrupted_dropped": n_bad, "n_filled": m}


def main():
    gallery = load_gallery()
    if gallery is None:
        print("NO GALLERY"); return
    base = torch.load(U.work_path("outputs_final", "submission_current_best.pt"), weights_only=False)
    out_dir = U.work_path("reports", "build", "headleak")
    U.ensure_dir(out_dir)
    mean, Eig = eigenbasis(gallery, k=120)                  # CelebA face subspace for garbage filter
    report = {}
    cand = {k: base[k].float().clamp(0, 1).clone() for k in U.MODELS}
    for name in ["model9", "model11"]:
        slot, info = run_model(name, gallery, base, out_dir, mean, Eig)
        cand[name] = slot
        report[name] = info
        print(f"  {name}: decoder_val_ssim={info['decoder_val_ssim']} filled {info['n_filled']}/128", flush=True)
    errs = validate_submission(cand)
    if errs:
        print("INVALID:", errs[:3])
    else:
        torch.save({k: cand[k].to(torch.float32).contiguous() for k in U.MODELS},
                   U.work_path("outputs_final", "merge_headleak_vit.pt"))
        print("wrote outputs_final/merge_headleak_vit.pt (ViT 9/11 = decoded singleton faces)")
    U.save_json(report, U.work_path("reports", "build", "headleak_report.json"))
    print("\nGT-FREE VERDICT: decoder_val_ssim is the honest signal. If >~0.45 the embedding decodes to")
    print("recognizable faces -> real per-sample ViT signal (submit). If low, LN stripped too much.")


if __name__ == "__main__":
    main()
