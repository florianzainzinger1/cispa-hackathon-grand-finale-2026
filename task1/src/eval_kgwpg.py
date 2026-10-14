"""Evaluate exact-KGW-null conditioning (kgwpg mode) vs champion.

Loads cache/kgwpg_{split}.npz (cluster top-k extraction), attaches kgw_pgreen
to the signal frames on the fly, evaluates clean fit-train/eval-val for
several level-edge sets. Also reports kgw-dominant recall.
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

def load_with_pg(split):
    df = pd.read_parquet(C / f"{split}_signals.parquet")
    d = np.load(C / f"kgwpg_{split}.npz")
    assert len(d["kgw_pgreen"]) == len(df), f"{split}: length mismatch"
    df = df.copy()
    df["kgw_pgreen"] = d["kgw_pgreen"]
    return df

tr = load_with_pg("train")
tre = pd.read_parquet(C / "train_evidence.parquet")
va = load_with_pg("validation")
vae = pd.read_parquet(C / "validation_evidence.parquet")
y = va["label"].to_numpy(); wm = y == 1
dom = vae[SCHEMES].to_numpy().argmax(axis=1)
kgw_dom = wm & (dom == 1)

# distribution sanity
pg = va["kgw_pgreen"].to_numpy()
ok = ~np.isnan(pg)
print(f"val kgw_pgreen: n={ok.sum()} mean={pg[ok].mean():.4f} std={pg[ok].std():.4f}")
print(f"  quantiles 1/5/25/50/75/95/99: "
      + " ".join(f"{np.quantile(pg[ok],q):.3f}" for q in [.01,.05,.25,.5,.75,.95,.99]))

BEST = {"unigram": "entropy", "kgw": "entropy", "gumbel": "logprob", "textseal": "logprob"}

def run(tag, cond, edges=None, pe=0.03, ms=80.0):
    if edges is not None:
        hmm.KGWPG_EDGES = np.array(edges)
    params = fit_emissions(tr, tre, use_entropy=cond)
    logA, logpi = make_transitions(pe, ms)
    sc = hmm_scores(va, params, logA, logpi)
    t01 = tpr_at_fpr(y, sc, 0.001)
    tau = np.quantile(sc[y == 0], 1 - 0.001)
    krec = (kgw_dom & (sc >= tau)).sum() / kgw_dom.sum()
    print(f"  {t01:.4f}  kgw-recall {krec:.3f}   {tag}")
    return t01

print(f"\n{'T@0.1%':>8s}  config   (v2 cache, trans 0.03/80)")
run("champion (kgw=entropy) [expect 0.4384]", BEST)
run("kgw=kgwpg edges [.05,.15,.35,.55]", {**BEST, "kgw": "kgwpg"}, [.05, .15, .35, .55])
run("kgw=kgwpg edges [.10,.20,.30,.40]", {**BEST, "kgw": "kgwpg"}, [.10, .20, .30, .40])
run("kgw=kgwpg edges [.02,.10,.25,.45,.65]", {**BEST, "kgw": "kgwpg"}, [.02, .10, .25, .45, .65])
run("kgw=kgwpg edges [.15,.35]", {**BEST, "kgw": "kgwpg"}, [.15, .35])
run("kgw=kgwpg edges [.02,.08,.20,.50,.80] (bimodal)", {**BEST, "kgw": "kgwpg"}, [.02, .08, .20, .50, .80])
run("kgw=kgwpg_ent (joint pg x entropy)", {**BEST, "kgw": "kgwpg_ent"})
