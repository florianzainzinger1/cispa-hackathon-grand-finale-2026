"""kgwpg plateau mapping: finer edge sets, joint kgwpg x entropy, transitions.

Clean fit-train/eval-val. Baseline champion 0.4384 (trans 0.03/80); best from
eval_kgwpg.py: edges [.02,.10,.25,.45,.65] -> 0.4427.
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

BEST = {"unigram": "entropy", "kgw": "entropy", "gumbel": "logprob", "textseal": "logprob"}

def run(tag, cond, edges=None, pe=0.03, ms=80.0, coarse=None, ent_c=None):
    if edges is not None:
        hmm.KGWPG_EDGES = np.array(edges)
    if coarse is not None:
        hmm.KGWPG_EDGES_C = np.array(coarse)
    if ent_c is not None:
        hmm.KGWPG_ENT_EDGES_C = np.array(ent_c)
    params = fit_emissions(tr, tre, use_entropy=cond)
    logA, logpi = make_transitions(pe, ms)
    sc = hmm_scores(va, params, logA, logpi)
    t01 = tpr_at_fpr(y, sc, 0.001)
    tau = np.quantile(sc[y == 0], 1 - 0.001)
    krec = (kgw_dom & (sc >= tau)).sum() / kgw_dom.sum()
    print(f"  {t01:.4f}  kgw-recall {krec:.3f}   {tag}", flush=True)
    return t01

KP = {**BEST, "kgw": "kgwpg"}
print(f"{'T@0.1%':>8s}  config")
run("champion kgw=entropy 0.03/80", BEST)
run("kgwpg [.02,.10,.25,.45,.65] 0.03/80 (best so far)", KP, [.02, .10, .25, .45, .65])
# finer low end (where the LR is largest: green at tiny pgreen)
run("kgwpg [.01,.05,.15,.35,.60]", KP, [.01, .05, .15, .35, .60])
run("kgwpg [.005,.03,.12,.30,.55,.80]", KP, [.005, .03, .12, .30, .55, .80])
run("kgwpg [.01,.04,.10,.20,.35,.55,.80]", KP, [.01, .04, .10, .20, .35, .55, .80])
run("kgwpg [.02,.10,.25,.45,.65,.85]", KP, [.02, .10, .25, .45, .65, .85])
# joint pgreen x entropy
KJ = {**BEST, "kgw": "kgwpg_ent"}
run("kgwpg_ent pg[.02,.35] x ent[1.0]", KJ)
run("kgwpg_ent pg[.02,.35] x ent[0.5,2.0]", KJ, ent_c=[0.5, 2.0])
run("kgwpg_ent pg[.05,.55] x ent[1.0]", KJ, coarse=[.05, .55], ent_c=[1.0])
# transitions cross-check on the best edge set
run("kgwpg [.02,.10,.25,.45,.65] 0.02/45", KP, [.02, .10, .25, .45, .65], pe=0.02, ms=45.0)
run("kgwpg [.02,.10,.25,.45,.65] 0.04/100", KP, [.02, .10, .25, .45, .65], pe=0.04, ms=100.0)
