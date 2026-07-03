"""READ-ONLY probe 2 (agent3): do models that share a classifier head also share
their 128 private images? model6 & model12 have IDENTICAL heads; model6 is our worst
model (capture 0.19) and model12 is clean (0.99). If they share images, model12's clean
recon can rebuild model6. Definitive test: pass a source model's recon images through
TARGET's conv and measure the residual against TARGET's fc1 gradient rowspace (the true
inputs give low residual; unrelated faces give high). Touches no pipeline/submission.
"""
import os
import torch
import torch.nn.functional as F

import utils as U
import models as M
import attack_analytic as AA


def fc1_rowspace(name, k=128):
    grad = U.load_grad(name)
    state = U.load_state(name)
    g = grad["gradients"]
    base = AA.first_leaky_linear(name, state, g)
    gW = g[base + ".weight"].float()
    _, S, Vh = torch.linalg.svd(gW, full_matrices=False)
    return Vh[:k].contiguous(), grad


def feat_of(name, imgs):
    grad = U.load_grad(name)
    state = U.load_state(name)
    meta = {"family": "cnn", "activation": grad["activation"], "feature_shape": grad["feature_shape"]}
    m = M.build_model(name, meta, state)
    with torch.no_grad():
        f = m.conv_feature(imgs.float())
    return f.reshape(f.shape[0], -1)


def resid_median(V, X):
    Xc = X - X.mean(dim=1, keepdim=True)          # DC out
    coeff = Xc @ V.t()
    cap = (coeff ** 2).sum(dim=1)
    tot = (Xc ** 2).sum(dim=1).clamp_min(1e-9)
    return float((1 - cap / tot).clamp_min(0).sqrt().median())


def main():
    sub = torch.load(U.work_path("outputs_v3", "submission_v3.pt"), weights_only=False)
    torch.manual_seed(0)
    rand = torch.rand(128, 3, 64, 64)
    # all stride-1 CNNs share feature dim; test which recon best explains each target's fc1 grad
    targets = ["model6", "model12", "model10"]
    sources = ["model6", "model10", "model12", "model2", "RANDOM"]
    print(f"{'target':8} | " + " ".join(f"{s:>8}" for s in sources))
    print("(median fc1-rowspace residual of SOURCE's recon imgs through TARGET's conv; low=explains target)")
    rep = {}
    for tgt in targets:
        V, grad = fc1_rowspace(tgt)
        row = {}
        cells = []
        for src in sources:
            imgs = rand if src == "RANDOM" else sub[src].float()
            try:
                Xf = feat_of(tgt, imgs)
                r = resid_median(V, Xf)
            except Exception as e:
                r = float("nan")
            row[src] = r
            cells.append(f"{r:8.4f}")
        rep[tgt] = row
        print(f"{tgt:8} | " + " ".join(cells))
    U.save_json(rep, U.work_path("reports", "build", "verify_probe2.json"))
    print("\nInterpretation: if SOURCE(model12) residual ~ TARGET(model6) self-residual and << RANDOM,")
    print("the two models likely SHARE images -> model12's clean recon can rebuild model6.")
    print("wrote reports/build/verify_probe2.json")


if __name__ == "__main__":
    main()
