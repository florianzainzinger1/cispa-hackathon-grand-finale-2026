"""Boundary-bleed rescue: HMM as region proposer + WaterSeeker-style local
interval refinement + soft boundary down-weighting.

Mechanism target (report 15): 94.1% of the 51-token FP budget lies within 10
tokens of true spans (posterior exit-tail). Oracle headroom +0.068 val.

Pipeline per document:
  1. champion HMM posterior (fit train, cond best, trans 0.03/80) -> base score
  2. span proposals: contiguous runs of P(wm) > 0.5
  3. per proposal: dominant WM state s*; evidence ev = ll[:,s*] - ll[:,0]
  4. local window [s-R, e+R): argmax interval statistic sum(ev)/sqrt(len)
     over subsegments (min length Lmin) -> refined core
  5. rescore: weight 1 in refined core, alpha in window remainder,
     w_out elsewhere (w_out=1 -> surgical variant; w_out=0 -> hard variant)

Eval: pooled TPR@0.1%FPR + boundary diagnostics (FP proximity, edge/core
recall, per-scheme recall, top FP docs).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hmm import (fit_emissions, make_transitions, emission_loglik,  # noqa
                 forward_backward, SCHEMES)
from metrics import tpr_at_fpr  # noqa

C = Path(__file__).resolve().parent.parent / "cache"

BEST = {"unigram": "entropy", "kgw": "entropy", "gumbel": "logprob", "textseal": "logprob"}

tr = pd.read_parquet(C / "train_signals.parquet")
tre = pd.read_parquet(C / "train_evidence.parquet")
va = pd.read_parquet(C / "validation_signals.parquet")
vae = pd.read_parquet(C / "validation_evidence.parquet")
y = va["label"].to_numpy()
docs = va["document_id"].to_numpy()

params = fit_emissions(tr, tre, use_entropy=BEST)
logA, logpi = make_transitions(0.03, 80.0)
ll_all = emission_loglik(va, params)

# per-doc posterior + doc boundaries
doc_slices = []
start = 0
for i in range(1, len(va) + 1):
    if i == len(va) or docs[i] != docs[start]:
        doc_slices.append((start, i))
        start = i

post_wm = np.zeros(len(va))
sc_base = np.zeros(len(va))
for a, b in doc_slices:
    post = forward_backward(ll_all[a:b], logA, logpi)
    p_clean = np.clip(post[:, 0], 1e-30, 1.0)
    sc_base[a:b] = -np.log10(p_clean)
    post_wm[a:b] = 1.0 - post[:, 0]

base_tpr = tpr_at_fpr(y, sc_base, 0.001)
print(f"champion base: {base_tpr:.4f}  (expect 0.4384)")


def proposals(a, b, thr=0.5):
    """Contiguous runs of post_wm > thr within doc slice [a,b)."""
    runs = []
    m = post_wm[a:b] > thr
    i = 0
    n = b - a
    while i < n:
        if m[i]:
            j = i
            while j < n and m[j]:
                j += 1
            runs.append((a + i, a + j))
            i = j
        else:
            i += 1
    return runs


def refine(ev, lo, hi, lmin):
    """Best subsegment [i,j) of ev[lo:hi] by sum/sqrt(len); global indices."""
    w = ev[lo:hi]
    n = len(w)
    if n < lmin:
        return lo, hi
    P = np.concatenate([[0.0], np.cumsum(w)])
    best_stat, best = -np.inf, (0, n)
    for i in range(0, n - lmin + 1):
        lens = np.arange(lmin, n - i + 1, dtype=np.float64)
        sums = P[i + lmin: n + 1] - P[i]
        stats = sums / np.sqrt(lens)
        k = int(stats.argmax())
        if stats[k] > best_stat:
            best_stat, best = float(stats[k]), (i, i + lmin + k)
    return lo + best[0], lo + best[1]


def build_weights(R, lmin, alpha, w_out, thr=0.5):
    """Two passes: boundary zones (window minus refined core) -> alpha,
    refined cores -> 1. Outside all windows -> w_out."""
    w = np.full(len(va), w_out, dtype=np.float64)
    cores = []
    n_prop = 0
    for a, b in doc_slices:
        for s, e in proposals(a, b, thr):
            n_prop += 1
            lo, hi = max(a, s - R), min(b, e + R)
            seg = ll_all[s:e]
            s_star = 1 + int((seg[:, 1:5] - seg[:, [0]]).sum(axis=0).argmax())
            ev = ll_all[:, s_star] - ll_all[:, 0]
            ri, rj = refine(ev, lo, hi, lmin)
            w[lo:hi] = np.minimum(w[lo:hi], alpha)
            cores.append((ri, rj))
    for ri, rj in cores:
        w[ri:rj] = 1.0
    return w, n_prop


def erosion_weights(k, alpha, thr=0.5):
    """Down-weight k tokens at each PROPOSAL edge (posterior exit tail);
    proposal core and everything outside keep weight 1."""
    w = np.ones(len(va), dtype=np.float64)
    cores = []
    for a, b in doc_slices:
        for s, e in proposals(a, b, thr):
            w[s:min(s + k, e)] = np.minimum(w[s:min(s + k, e)], alpha)
            w[max(e - k, s):e] = np.minimum(w[max(e - k, s):e], alpha)
            if e - s > 2 * k:
                cores.append((s + k, e - k))
    for ri, rj in cores:
        w[ri:rj] = 1.0
    return w


# --- diagnostics helpers -------------------------------------------------
dom = vae[SCHEMES].to_numpy().argmax(axis=1)
wm = y == 1

# distance of each token to nearest true-span edge (inside=+, outside=-)
dist_out = np.full(len(va), 10**9)   # clean: distance to nearest span
depth_in = np.full(len(va), 10**9)   # wm: distance to own span edge
for a, b in doc_slices:
    yy = y[a:b]
    idx = np.arange(b - a)
    spans = []
    i = 0
    while i < b - a:
        if yy[i]:
            j = i
            while j < b - a and yy[j]:
                j += 1
            spans.append((i, j))
            i = j
        else:
            i += 1
    for i in idx:
        if yy[i]:
            for s, e in spans:
                if s <= i < e:
                    depth_in[a + i] = min(i - s, e - 1 - i)
                    break
        else:
            dmin = 10**9
            for s, e in spans:
                dmin = min(dmin, abs(i - s) if i < s else i - e + 1)
            dist_out[a + i] = dmin


def report(tag, sc):
    t = tpr_at_fpr(y, sc, 0.001)
    clean = ~wm
    tau = np.quantile(sc[clean], 1 - 0.001)
    fp = clean & (sc >= tau)
    n_fp = int(fp.sum())
    prox = {k: int((fp & (dist_out <= k)).sum()) for k in (3, 5, 10, 20)}
    hit = sc >= tau
    edge = {k: float(hit[wm & (depth_in <= k)].mean()) for k in (3, 5, 10)}
    core = float(hit[wm & (depth_in > 10)].mean())
    srec = {s: float(hit[wm & (dom == i)].mean()) for i, s in enumerate(SCHEMES)}
    top_docs = pd.Series(docs[fp]).value_counts().head(3).to_dict()
    print(f"{tag:34s} {t:.4f} (d{t-base_tpr:+.4f}) | FP<=3/5/10/20: "
          f"{prox[3]}/{prox[5]}/{prox[10]}/{prox[20]} of {n_fp} | "
          f"edge<=3/5/10: {edge[3]:.2f}/{edge[5]:.2f}/{edge[10]:.2f} core {core:.2f} | "
          f"g/t/k/u {srec['gumbel']:.2f}/{srec['textseal']:.2f}/{srec['kgw']:.2f}/{srec['unigram']:.2f} | "
          f"topFP {top_docs}")
    return t


print("\n--- baseline diagnostics ---")
report("champion", sc_base)

print("\n--- surgical (w_out=1): only boundary zones down-weighted ---")
for R in (10, 15, 20, 30):
    for alpha in (0.0, 0.1, 0.25, 0.5):
        w, np_ = build_weights(R, 16, alpha, w_out=1.0)
        report(f"S R={R} Lmin=16 a={alpha}", sc_base * w)

print("\n--- proposal-edge erosion: alpha on k tokens at proposal edges ---")
for k in (2, 3, 5, 8):
    for alpha in (0.0, 0.1, 0.25, 0.5):
        w = erosion_weights(k, alpha)
        report(f"E k={k} a={alpha}", sc_base * w)
