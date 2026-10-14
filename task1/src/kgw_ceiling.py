"""Bound the achievable KGW gain with LOCAL proxies for p_green_base.

Measures overall val TPR@0.1%FPR AND kgw-dominant-scheme recall for several
KGW emission variants (rest of schemes held at champion). Tells us whether a
cluster trip for the exact kgw_pgreen is worth it.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hmm  # noqa
from hmm import fit_emissions, make_transitions, hmm_scores, SCHEMES  # noqa
from metrics import tpr_at_fpr  # noqa

C = Path(__file__).resolve().parent.parent / "cache"
tr = pd.read_parquet(C / "train_signals.parquet")
tre = pd.read_parquet(C / "train_evidence.parquet")
va = pd.read_parquet(C / "validation_signals.parquet")
vae = pd.read_parquet(C / "validation_evidence.parquet")
y = va["label"].to_numpy(); wm = y == 1
dom = vae[SCHEMES].to_numpy().argmax(axis=1)
kgw_dom = wm & (dom == 1)

BEST = {"unigram": "entropy", "kgw": "entropy", "gumbel": "logprob", "textseal": "logprob"}


def measure(kgw_mode, tag):
    cond = {**BEST, "kgw": kgw_mode}
    params = fit_emissions(tr, tre, use_entropy=cond)
    logA, logpi = make_transitions(0.02, 45.0)
    sc = hmm_scores(va, params, logA, logpi)
    overall = tpr_at_fpr(y, sc, 0.001)
    tau = np.quantile(sc[y == 0], 1 - 0.001)
    krec = ((kgw_dom) & (sc >= tau)).sum() / kgw_dom.sum()
    print(f"  overall {overall:.4f}   kgw-recall {krec:.3f}   {tag}")
    return overall, krec


print("KGW emission variants (rest = champion, trans 0.02/45):")
print(f"  {'overall':>8s} {'kgw-recall':>10s}   config")
measure("entropy", "entropy [current]")
measure("logprob", "logprob")
measure("rank", "rank")
measure("both", "both (ent x lp)")
measure(None, "none (unconditioned)")

# KGW-only detector CEILING: zero out all other emissions, measure kgw recall
print("\nKGW-ONLY detector (all other schemes muted) — the pure KGW ceiling:")
for mode in ["entropy", "logprob", "both", "rank"]:
    cond = {"unigram": None, "kgw": mode, "gumbel": None, "textseal": None}
    # mute others by making their emissions uninformative: fit but then we only
    # let kgw contribute. Easiest: build params with only kgw informative.
    params = fit_emissions(tr, tre, use_entropy={**BEST, "kgw": mode})
    for s in ["unigram", "gumbel", "textseal"]:
        params[s]["log_wm"] = params[s]["log_clean"].copy()  # LR=1 -> no signal
    logA, logpi = make_transitions(0.02, 45.0)
    sc = hmm_scores(va, params, logA, logpi)
    tau = np.quantile(sc[y == 0], 1 - 0.001)
    krec = ((kgw_dom) & (sc >= tau)).sum() / kgw_dom.sum()
    # among ALL wm (since only kgw fires, overall ~ kgw contribution)
    overall = tpr_at_fpr(y, sc, 0.001)
    print(f"  kgw={mode:8s}  kgw-recall {krec:.3f}  overall {overall:.4f}")
