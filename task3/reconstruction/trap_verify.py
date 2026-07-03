"""trap_verify.py  (Agent T / lever L2) — GT-free verification of recovered images.

We have no ground truth, so we score a recovered set by two analytic certificates:

  1. GRADIENT SELF-CONSISTENCY: re-feed the recovered images (with an inferred
     label multiset from infer_labels) through the actual model, backprop the same
     CE loss (mean reduction = averaged client gradient), and compare the produced
     gradient to the OBSERVED gradient by cosine — on the first (image) layer and
     over ALL parameters. A faithful reconstruction reproduces the gradient.

  2. SUBSPACE RESIDUAL: a true batch member lies in rowspace(gW) (gW = sum of 128
     rank-1 sample terms). Project each recovered image (MLP: pixel space; CNN:
     conv-feature space) onto the top-128 right-singular subspace of the observed
     first-layer weight-gradient and report the residual ||x-Px||/||x||. Low =
     plausible member. (NB: blends of >=2 members are ALSO in-subspace, so residual
     rules out garbage/empty-bin noise, not collisions — cosine self-consistency
     over the whole set is the collision-sensitive signal.)

Compares the trap recovery against the c3 baseline slice for the same model, so the
STATUS table can show whether differencing/ratio beats our analytic-ratio faces.

Run via sbatch job_trap.sh verify  (GPU).   python -u trap_verify.py [model...]
"""
import os
import sys
import json
import torch
import torch.nn.functional as F

import utils as U
import models as M
from infer_labels import assign_labels

_dev = os.environ.get("FL_DEVICE", "cpu")
DEVICE = "cuda" if (_dev == "cuda" and torch.cuda.is_available()) else "cpu"


def first_biased_linear(state, g):
    for pname, w in state.items():
        if pname.endswith(".weight") and w.ndim == 2:
            base = pname[:-7]
            if base + ".bias" in g and g[base + ".bias"] is not None:
                return base
    return None


def cos(a, b):
    return float(F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item())


def refeed_grad(model, imgs, labels):
    model.zero_grad(set_to_none=True)
    x = imgs.to(DEVICE)
    y = labels.to(DEVICE)
    logits = model(x)
    loss = F.cross_entropy(logits, y, reduction="mean")
    loss.backward()
    return {n: p.grad.detach().clone() for n, p in model.named_parameters()
            if p.grad is not None}


def subspace_residual(feats_flat, gW, k=128):
    """feats_flat: (n, d) rows to test; gW: (out, d) observed weight-grad.
    Residual of each row after projecting onto top-k right-singular subspace of gW."""
    gW = gW.float().to(DEVICE)
    _, _, Vh = torch.linalg.svd(gW, full_matrices=False)
    Vk = Vh[:k]                                  # (k, d)
    x = feats_flat.float().to(DEVICE)
    proj = (x @ Vk.t()) @ Vk                     # (n, d)
    res = (x - proj).norm(dim=1) / (x.norm(dim=1) + 1e-12)
    return res.cpu()


def verify_set(name, imgs, tag):
    """imgs: (n,3,64,64) in [0,1]. Returns metrics dict."""
    grad = U.load_grad(name)
    state = U.load_state(name)
    g = grad["gradients"]
    fam = grad.get("family")
    if fam == "vit":
        return {"tag": tag, "n": int(imgs.shape[0]), "note": "vit skip"}
    base = first_biased_linear(state, g)
    meta = {"family": fam, "activation": grad.get("activation"),
            "feature_shape": grad.get("feature_shape")}
    model = M.build_model(name, meta, state).to(DEVICE)
    for p in model.parameters():
        p.requires_grad_(True)

    n = imgs.shape[0]
    labels = assign_labels(name, N=n)[:n]
    rg = refeed_grad(model, imgs.clamp(0, 1), labels)

    # first-layer (image) cosine
    obsW = g[base + ".weight"].float().to(DEVICE)
    obsb = g[base + ".bias"].float().to(DEVICE)
    fl_cos = cos(rg[base + ".weight"], obsW)
    flb_cos = cos(rg[base + ".bias"], obsb)
    # all-parameter cosine (concat in observed-key order where both exist)
    a, b = [], []
    for k in g:
        if k in rg and g[k] is not None:
            a.append(g[k].reshape(-1).float().to(DEVICE))
            b.append(rg[k].reshape(-1))
    all_cos = cos(torch.cat(a), torch.cat(b)) if a else None

    # subspace residual
    if base.endswith("net.0") or obsW.shape[1] == U.IMG_NUMEL:   # MLP: pixel space
        feats = imgs.reshape(n, -1)
    else:                                                        # CNN: conv-feature space
        with torch.no_grad():
            feats = model.conv_feature(imgs.to(DEVICE)).reshape(n, -1).cpu()
    res = subspace_residual(feats, obsW)
    return {
        "tag": tag, "n": int(n),
        "firstlayer_W_cos": round(fl_cos, 4), "firstlayer_b_cos": round(flb_cos, 4),
        "all_param_cos": round(all_cos, 4) if all_cos is not None else None,
        "subspace_resid_median": round(float(res.median()), 4),
        "subspace_resid_p10": round(float(res.quantile(0.1)), 4),
        "subspace_resid_frac_lt_0.2": round(float((res < 0.2).float().mean()), 4),
    }


def load_candidate(name):
    p = U.work_path("outputs_trap", f"trap_{name}.pt")
    if not os.path.exists(p):
        return None
    return torch.load(p, weights_only=False)["images"].float()


def load_baseline(name):
    for rel in [("outputs_v3", "submission_v3.pt"),
                ("outputs_final", "submission_current_best.pt"),
                ("outputs", "submission.pt")]:
        p = U.work_path(*rel)
        if os.path.exists(p):
            sub = torch.load(p, weights_only=False)
            if name in sub:
                return sub[name].float(), rel[0]
    return None, None


def main():
    names = sys.argv[1:] or U.MODELS
    out = {}
    for name in names:
        rec = {}
        cand = load_candidate(name)
        if cand is not None and cand.shape[0] > 0:
            try:
                rec["trap"] = verify_set(name, cand, "trap")
            except Exception as e:
                rec["trap"] = {"error": str(e)}
        base, src = load_baseline(name)
        if base is not None:
            try:
                rec["baseline"] = verify_set(name, base, f"baseline:{src}")
            except Exception as e:
                rec["baseline"] = {"error": str(e)}
        out[name] = rec
        t = rec.get("trap", {}); bl = rec.get("baseline", {})
        print(f"{name:8s} trap[flW={t.get('firstlayer_W_cos')} all={t.get('all_param_cos')} "
              f"res<0.2={t.get('subspace_resid_frac_lt_0.2')} n={t.get('n')}]  "
              f"base[flW={bl.get('firstlayer_W_cos')} all={bl.get('all_param_cos')} "
              f"res<0.2={bl.get('subspace_resid_frac_lt_0.2')}]", flush=True)
    U.save_json(out, U.work_path("reports", "trap", "verify.json"))
    print("\nwrote", U.work_path("reports", "trap", "verify.json"))


if __name__ == "__main__":
    main()
