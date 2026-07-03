"""Cocktail-Party Attack (CPA) — batch source separation of the first-layer
gradient (agent3 / lever L3). Faithful to facebookresearch/cocktail_party_attack
with the corrections found in research (scale via per-image min-max AND an optional
SPEAR-style bias-scale fix; sign via low-TV orientation; negentropy off by default).

MLP (1,4,5,8): sources ARE the flattened images -> reshape (3,64,64).
CNN (6,10,12,2,7,3): CPA runs on the fc1 gradient -> 128 conv-feature maps, which
are then inverted through the first conv (reuse attack_conv_invert.invert_conv).

Whitening/objective (per research spec):
  center per-row, cov = Xzc Xzc^T/(d-1), top-128 eig -> K = D^-1/2 E^T, Xw = K Xzc.
  W_hat free (init I or analytic warm-start), row-normalized each step.
  loss = decor*mean(exp(T|cos(W)|)-1) + tv*TV(Sd)  [MLP]  or  + nv*signterm + l1*|Sd|  [CNN].

Heavy (25k Adam steps) -> run via sbatch on GPU. Env: FL_DEVICE, FL_BSS_ITERS.
Namespace: bss_*  ->  outputs_bss/.
"""
import os
import sys
import time
import torch
import torch.nn.functional as F

import utils as U
import bss_common as B
import attack_conv_invert as CI

DEVICE = os.environ.get("FL_DEVICE", "cpu")
if DEVICE == "cuda" and not torch.cuda.is_available():
    DEVICE = "cpu"
ITERS = int(os.environ.get("FL_BSS_ITERS", "20000"))


def tv_mean(x):
    dh = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    dw = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    return dh + dw


def analytic_warm_init(gW, gb, Xw, Bk, tau=1e-6):
    """Seed W_hat from Bk diverse analytic-ratio rows gW[j]/gb[j] (farthest-point
    in cosine distance), solved into the whitened frame: W = S_seed @ pinv(Xw)."""
    idx_pool = torch.where(gb.abs() > tau)[0]
    if idx_pool.numel() < Bk:
        idx_pool = torch.arange(gW.shape[0], device=gW.device)
    R = gW[idx_pool] / gb[idx_pool].unsqueeze(1)      # (P,d) analytic ratios
    Rc = R - R.mean(dim=1, keepdim=True)
    Rn = F.normalize(Rc, dim=1)
    G = (Rn @ Rn.t()).abs()                           # (P,P) pool-local cosine gram
    P = G.shape[0]
    start = int(Rc.abs().amax(dim=1).argmax())
    chosen = [start]
    maxsim = G[start].clone()
    maxsim[start] = 2.0
    while len(chosen) < min(Bk, P):
        nxt = int(maxsim.argmin())
        chosen.append(nxt)
        maxsim = torch.maximum(maxsim, G[nxt])
        maxsim[torch.tensor(chosen, device=G.device)] = 2.0
    S_seed = Rc[chosen]
    if S_seed.shape[0] < Bk:
        extra = torch.randn(Bk - S_seed.shape[0], S_seed.shape[1], device=gW.device)
        S_seed = torch.cat([S_seed, extra], 0)
    return S_seed @ torch.linalg.pinv(Xw)             # (Bk,Bk)


def cpa_unmix(Xw, K, mu, shape, iters, lr, tv, T, decor, ne, nv, l1,
              W_init=None, log_every=0):
    """Return de-centered raw sources Sd (Bk, d)."""
    Bk, d = Xw.shape
    W = (torch.eye(Bk, device=Xw.device) if W_init is None else W_init.clone())
    W = W.to(Xw.device).clone().requires_grad_(True)
    opt = torch.optim.Adam([W], lr=lr)
    Kmu = K @ mu                                     # (Bk,1) de-center offset
    for it in range(iters):
        opt.zero_grad()
        Wn = W / (W.norm(dim=1, keepdim=True) + 1e-12)
        S = Wn @ Xw
        Sd = S + Wn @ Kmu
        loss = torch.zeros((), device=Xw.device)
        if decor > 0:
            cosM = (Wn @ Wn.t()).abs()
            loss = loss + decor * (torch.exp(cosM * T) - 1.0).mean()
        if tv > 0 and shape is not None:
            loss = loss + tv * tv_mean(Sd.view(Bk, *shape))
        if ne > 0:
            J = ((torch.log(torch.cosh(S) + 1e-12).mean(dim=1)) ** 2).mean()
            loss = loss - ne * J
        if nv > 0:
            loss = loss + nv * torch.minimum(F.relu(-Sd).norm(dim=1), F.relu(Sd).norm(dim=1)).mean()
        if l1 > 0:
            loss = loss + l1 * Sd.abs().mean()
        loss.backward()
        opt.step()
        if log_every and (it % log_every == 0 or it == iters - 1):
            print(f"      it {it:5d} loss={loss.item():.4f}", flush=True)
    with torch.no_grad():
        Wn = W / (W.norm(dim=1, keepdim=True) + 1e-12)
        Sd = (Wn @ Xw) + Wn @ Kmu
    return Sd.detach()


# --------------------------------------------------------------------------
# MLP path: sources are images
# --------------------------------------------------------------------------
MLP_SWEEP = [
    dict(tag="ref", tv=3.1, T=12.4, decor=1.47, warm=True),
    dict(tag="ref_cold", tv=3.1, T=12.4, decor=1.47, warm=False),
    dict(tag="lowT", tv=3.1, T=7.0, decor=1.47, warm=True),
    dict(tag="lowTV", tv=1.0, T=8.0, decor=2.0, warm=True),
    dict(tag="hidecor", tv=2.0, T=10.0, decor=3.0, warm=True),
    dict(tag="ne_on", tv=2.0, T=8.0, decor=1.5, warm=True, ne=0.2),
]


def map_sources_mlp(lay, Sd):
    """Two mappings from raw sources Sd (128,12288) to [0,1] images; return dict."""
    B128 = Sd.shape[0]
    S_img = Sd.view(B128, B.IMG_C, B.IMG_H, B.IMG_W)
    out = {}
    out["minmax"] = B.resolve_sign_lowTV(S_img)                       # ref-faithful
    Xsc, dsc, ok = B.biasscale_fix(lay["gW"].cpu(), lay["gb"].cpu(), Sd.cpu())
    out["biasscale"] = B.minmax_rows(Xsc.view(B128, B.IMG_C, B.IMG_H, B.IMG_W))
    return out


def run_mlp(name, iters, sweep, out_root):
    print(f"\n=== CPA MLP {name} (iters={iters}, device={DEVICE}) ===", flush=True)
    lay = B.load_layer(name, device=DEVICE)
    Xw, K, mu = B.whiten_topk(lay["gW"], B.N)
    Vbasis, Sbasis = None, None
    warm_cache = {}
    results = []
    best = None
    mdir = os.path.join(out_root, name)
    U.ensure_dir(mdir)
    for cfg in sweep:
        t0 = time.time()
        W_init = None
        if cfg.get("warm", True):
            if "w" not in warm_cache:
                warm_cache["w"] = analytic_warm_init(lay["gW"], lay["gb"], Xw, B.N)
            W_init = warm_cache["w"]
        Sd = cpa_unmix(Xw, K, mu, shape=(B.IMG_C, B.IMG_H, B.IMG_W), iters=iters,
                       lr=1e-3, tv=cfg.get("tv", 3.1), T=cfg.get("T", 12.4),
                       decor=cfg.get("decor", 1.47), ne=cfg.get("ne", 0.0),
                       nv=0.0, l1=0.0, W_init=W_init, log_every=max(1, iters // 4))
        maps = map_sources_mlp(lay, Sd)
        for mtag, X in maps.items():
            m, Vbasis, Sbasis = B.measure(lay, X, Vbasis, Sbasis)
            score = B.combined_score(m)
            row = {"cfg": cfg["tag"], "map": mtag, "score": round(score, 4), **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}}
            results.append(row)
            if best is None or score > best["score"]:
                best = {"score": score, "X": X.clone(), "cfg": cfg["tag"], "map": mtag}
            U.save_image_grid(X[:64], os.path.join(mdir, f"grid_{cfg['tag']}_{mtag}.png"))
        print(f"  cfg={cfg['tag']:8s} best_map_score seen; {time.time()-t0:.0f}s", flush=True)
    _finalize(name, lay, best, results, mdir, out_root)


# --------------------------------------------------------------------------
# CNN path: sources are conv features -> invert conv
# --------------------------------------------------------------------------
CNN_SWEEP = [
    dict(tag="cnn_ref", nv=0.13, l1=5.0, T=7.7, decor=5.3, warm=False),
    dict(tag="cnn_l1lo", nv=0.13, l1=1.0, T=7.7, decor=3.0, warm=False),
    dict(tag="cnn_warm", nv=0.13, l1=5.0, T=7.7, decor=5.3, warm=True),
]


def run_cnn(name, iters, sweep, out_root):
    print(f"\n=== CPA CNN {name} (iters={iters}, device={DEVICE}) ===", flush=True)
    lay = B.load_layer(name, device=DEVICE)
    C, H, Wd = lay["feature_shape"]
    Xw, K, mu = B.whiten_topk(lay["gW"], B.N)
    model = B.build_cnn(lay).to(DEVICE)
    strided = model.stride > 1
    base_res = 64 if not strided else H
    act = lay["activation"]
    Vbasis, Sbasis = None, None
    warm_cache = {}
    results = []
    best = None
    mdir = os.path.join(out_root, name)
    U.ensure_dir(mdir)
    for cfg in sweep:
        t0 = time.time()
        W_init = None
        if cfg.get("warm", False):
            if "w" not in warm_cache:
                warm_cache["w"] = analytic_warm_init(lay["gW"], lay["gb"], Xw, B.N)
            W_init = warm_cache["w"]
        Sd = cpa_unmix(Xw, K, mu, shape=(C, H, Wd), iters=iters, lr=1e-3,
                       tv=0.0, T=cfg["T"], decor=cfg["decor"], ne=0.0,
                       nv=cfg["nv"], l1=cfg["l1"], W_init=W_init, log_every=max(1, iters // 4))
        # bias-scale fix -> correctly scaled feature maps, then match activation range
        Xsc, dsc, ok = B.biasscale_fix(lay["gW"].cpu(), lay["gb"].cpu(), Sd.cpu())
        feat = Xsc.view(B.N, C, H, Wd)
        if act == "relu":
            feat = feat.clamp_min(0.0)
        elif act == "sigmoid":
            feat = feat.clamp(0.0, 1.0)
        elif act == "tanh":
            feat = feat.clamp(-1.0, 1.0)
        rec = CI.invert_conv(model, feat.to(DEVICE), steps=400,
                             lr=0.03 if strided else 0.05,
                             tv=5e-3 if strided else 1e-4,
                             device=DEVICE, base_res=base_res)
        X = rec.cpu()
        m, Vbasis, Sbasis = B.measure(lay, X, Vbasis, Sbasis)
        score = B.combined_score(m)
        results.append({"cfg": cfg["tag"], "map": "convinv", "score": round(score, 4),
                        **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}})
        if best is None or score > best["score"]:
            best = {"score": score, "X": X.clone(), "cfg": cfg["tag"], "map": "convinv"}
        U.save_image_grid(X[:64], os.path.join(mdir, f"grid_{cfg['tag']}.png"))
        print(f"  cfg={cfg['tag']:10s} score={score:.4f}  {time.time()-t0:.0f}s", flush=True)
    _finalize(name, lay, best, results, mdir, out_root)


def _finalize(name, lay, best, results, mdir, out_root):
    results = sorted(results, key=lambda r: -r["score"])
    U.save_json({"name": name, "family": lay["family"], "activation": lay["activation"],
                 "best": {"cfg": best["cfg"], "map": best["map"], "score": best["score"]},
                 "results": results},
                os.path.join(mdir, "sweep.json"))
    torch.save({"images": best["X"].float().clamp(0, 1), "method": "cpa",
                "cfg": best["cfg"], "map": best["map"], "score": best["score"]},
               os.path.join(out_root, f"bss_{name}.pt"))
    U.save_image_grid(best["X"][:128], os.path.join(mdir, "grid_BEST.png"))
    print(f"  -> BEST {name}: cfg={best['cfg']} map={best['map']} score={best['score']:.4f}", flush=True)
    print(f"     top3: {[ (r['cfg'], r['map'], r['score']) for r in results[:3] ]}", flush=True)


def main():
    args = sys.argv[1:]
    names = [a for a in args if a.startswith("model")] or B.MLP_MODELS
    out_root = U.work_path("outputs_bss")
    U.ensure_dir(out_root)
    for name in names:
        lay_fam = B.load_layer(name)["family"]
        try:
            if lay_fam == "mlp":
                run_mlp(name, ITERS, MLP_SWEEP, out_root)
            elif lay_fam == "cnn":
                run_cnn(name, min(ITERS, 15000), CNN_SWEEP, out_root)
            else:
                print(f"{name}: family {lay_fam} not handled by CPA")
        except Exception as e:
            import traceback
            print(f"{name}: ERROR {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
