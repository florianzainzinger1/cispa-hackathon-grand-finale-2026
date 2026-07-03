"""Agent B — L0: iLRG per-class COUNT recovery (RESEARCH_FINDINGS §3-L0).

For each model, recover the exact per-class image COUNTS k_j (sum=128) from the
200-way classifier head gradient, using only the white-box head + its gradient.
Unlike plain label-SET recovery this gives the number of DISTINCT identities and
the multiplicity of each — which feeds:
  * per-label Hungarian blocks (match within a class),
  * L5 duplicate-averaging (a count-k class -> average its k slots for a sqrt(k)
    denoise, then submit k copies),
  * class-conditional priors for the other levers.

Method (iLRG, Ma et al. ICLR'23):
  e_i = grad_W_i / grad_b_i                        # per-class avg embedding (scale-free)
  p_i = softmax(alpha * (W @ e_i + b))             # white-box head re-scoring; sweep alpha
  solve the (C+1) x C system for counts k by pseudoinverse (closure sum k = K).
Pick alpha in {0.003,0.01,0.03} whose raw counts sum closest to K=128 (free self-check).

Login-node safe. Writes reports/build/label_counts.json + prints a per-model summary.
"""
import os
import sys
import json
import numpy as np
import torch

import utils as U
import infer_labels as IL

K = 128
ALPHAS = (0.003, 0.01, 0.03)


def head_wb_and_grad(name):
    """Return (W, b, gW, gb) for the classifier head: model weights + their grads."""
    grad = U.load_grad(name)
    state = U.load_state(name)
    g = grad["gradients"]
    wkey, bkey = IL.head_keys(state, g)
    if wkey is None or g.get(wkey) is None or g.get(bkey) is None:
        return None
    W = state[wkey].float().numpy()          # (C, m)
    b = state[bkey].float().numpy()          # (C,)
    gW = g[wkey].float().numpy()             # (C, m)
    gb = g[bkey].float().numpy()             # (C,)
    return W, b, gW, gb


def class_embeddings(gW, gb, clamp=1e-8):
    """e_i = gW_i / gb_i  (row i divided by scalar bias-grad i). Scale-free."""
    denom = gb.copy()
    denom[np.abs(denom) < clamp] = clamp * np.sign(denom[np.abs(denom) < clamp] + 1e-30)
    denom[denom == 0] = clamp
    return gW / denom[:, None]               # (C, m)


def probs_matrix(W, b, E, alpha):
    """probs[j] = softmax(alpha*(W @ e_j + b)); returns (C, C) with probs[j][i]."""
    logits = alpha * (E @ W.T + b[None, :])  # (C, C): row j = scores of embedding j
    logits = logits - logits.max(axis=1, keepdims=True)
    ex = np.exp(np.clip(logits, -60, 60))
    return ex / (ex.sum(axis=1, keepdims=True) + 1e-30)


def ilrg_counts(probs, grad_b, C=200, K=128):
    """(C+1)xC pseudoinverse solve for integer per-class counts summing to K.
    probs[j][i] = P(embedding j -> class i)."""
    coefs, vals = [[1.0] * C], [float(K)]                    # closure sum k = K
    for i in range(C):
        row = [probs[j][i] for j in range(C)]
        row[i] -= 1.0
        coefs.append(row)
        vals.append(K * grad_b[i])
    raw = np.linalg.pinv(np.array(coefs)).dot(np.array(vals))
    k = np.round(np.clip(raw, 0, K)).astype(int)
    d = int(k.sum() - K)
    err = k - raw
    if d > 0:
        k[np.argsort(-err)[:d]] -= 1
    elif d < 0:
        k[np.argsort(err)[:-d]] += 1
    k = np.clip(k, 0, K)
    return k, np.where(k > 0)[0], raw


def recover(name, C=200):
    hw = head_wb_and_grad(name)
    if hw is None:
        return {"name": name, "ok": False, "reason": "no head grad"}
    W, b, gW, gb = hw
    C = gb.shape[0]
    E = class_embeddings(gW, gb)
    best = None
    for a in ALPHAS:
        P = probs_matrix(W, b, E, a)
        k, present, raw = ilrg_counts(P, gb, C=C, K=K)
        raw_sum = float(np.clip(raw, 0, K).sum())
        score = abs(raw_sum - K)                              # self-consistency
        cand = {"alpha": a, "counts": k, "present": present, "raw_sum": raw_sum,
                "score": score}
        if best is None or score < best["score"]:
            best = cand
    k = best["counts"]
    present = best["present"]
    return {
        "name": name, "ok": True, "alpha": best["alpha"],
        "n_distinct": int((k > 0).sum()),
        "count_sum": int(k.sum()),
        "raw_sum_at_best_alpha": round(best["raw_sum"], 2),
        "max_count": int(k.max()),
        "n_dup_classes": int((k > 1).sum()),
        "top_classes": present[np.argsort(-k[present])][:15].tolist(),
        "top_counts": np.sort(k[present])[::-1][:15].tolist(),
        "counts": k.tolist(),
    }


def main():
    names = sys.argv[1:] or U.MODELS
    out = {}
    print(f"{'model':8} {'ok':>3} {'alpha':>6} {'distinct':>9} {'sum':>4} {'dupcls':>7} {'maxk':>5}")
    for name in names:
        r = recover(name)
        out[name] = r
        if r["ok"]:
            print(f"{name:8} {'y':>3} {r['alpha']:6.3f} {r['n_distinct']:9d} "
                  f"{r['count_sum']:4d} {r['n_dup_classes']:7d} {r['max_count']:5d}")
        else:
            print(f"{name:8} {'n':>3}  ({r['reason']})")
    U.save_json(out, U.work_path("reports", "build", "label_counts.json"))
    print("\nwrote", U.work_path("reports", "build", "label_counts.json"))


if __name__ == "__main__":
    main()
