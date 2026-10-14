"""Diagnose WHERE recovery fails at the 0.1%-FPR operating point.

Breaks recovered vs missed watermark tokens down by (a) dominant scheme
(argmax of window evidence) and (b) token logprob band. Reveals which
information source is untapped -> where a structural lever would pay.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hmm import fit_emissions, make_transitions, hmm_scores, SCHEMES  # noqa
from metrics import tpr_at_fpr  # noqa

C = Path(__file__).resolve().parent.parent / "cache"
tr = pd.read_parquet(C / "train_signals.parquet")
tre = pd.read_parquet(C / "train_evidence.parquet")
va = pd.read_parquet(C / "validation_signals.parquet")
vae = pd.read_parquet(C / "validation_evidence.parquet")
y = va["label"].to_numpy()

BEST = {"unigram": "entropy", "kgw": "entropy", "gumbel": "logprob", "textseal": "logprob"}
params = fit_emissions(tr, tre, use_entropy=BEST)
logA, logpi = make_transitions(0.02, 45.0)
sc = hmm_scores(va, params, logA, logpi)

# global 0.1% FPR threshold from clean tokens
clean = sc[y == 0]
tau = np.quantile(clean, 1 - 0.001)
print(f"Overall TPR@0.1%FPR = {tpr_at_fpr(y, sc, 0.001):.4f}  (tau={tau:.3f})")
print(f"clean={int((y==0).sum())}  wm={int((y==1).sum())}\n")

wm = y == 1
recovered = wm & (sc >= tau)

# per-scheme via dominant window evidence at each token
ev = vae[SCHEMES].to_numpy()
dom = ev.argmax(axis=1)
print("== recovery by dominant scheme (evidence argmax) ==")
print(f"  {'scheme':10s} {'wm_tok':>7s} {'recov':>7s} {'recall':>7s} {'mean_ev':>8s}")
for si, s in enumerate(SCHEMES):
    sel = wm & (dom == si)
    n = int(sel.sum())
    rec = int((sel & recovered).sum())
    r = rec / n if n else 0.0
    print(f"  {s:10s} {n:7d} {rec:7d} {r:7.3f} {ev[sel, si].mean() if n else 0:8.3f}")

# per-logprob band (watermark bias shrinks as logprob->0)
print("\n== recovery by token logprob band (wm tokens) ==")
lp = va["qwen_logprob"].to_numpy()
edges = [-np.inf, -3, -2, -1, -0.5, -0.2, 0.01]
for a, b in zip(edges[:-1], edges[1:]):
    sel = wm & (lp >= a) & (lp < b)
    n = int(sel.sum()); rec = int((sel & recovered).sum())
    print(f"  lp[{a:5.1f},{b:5.1f})  wm={n:6d}  recov={rec:6d}  recall={rec/n if n else 0:6.3f}")

# per-entropy band
print("\n== recovery by token entropy band (wm tokens) ==")
en = va["qwen_entropy"].to_numpy()
eedges = [0, 0.3, 1.0, 2.0, 100]
for a, b in zip(eedges[:-1], eedges[1:]):
    sel = wm & (en >= a) & (en < b)
    n = int(sel.sum()); rec = int((sel & recovered).sum())
    print(f"  ent[{a:5.1f},{b:6.1f})  wm={n:6d}  recov={rec:6d}  recall={rec/n if n else 0:6.3f}")

# how much would a PERFECT per-scheme detector add? upper bound per scheme:
# if we could recover ALL tokens of scheme X at 0 FPR cost, what's max TPR?
print("\n== ceiling: fraction of wm tokens per scheme (max recoverable if scheme cracked) ==")
for si, s in enumerate(SCHEMES):
    frac = (wm & (dom == si)).sum() / wm.sum()
    print(f"  {s:10s} {frac:6.3f} of all wm tokens")
