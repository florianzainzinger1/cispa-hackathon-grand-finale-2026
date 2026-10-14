"""First read on kgwpg BEFORE validation extraction lands: train split-half.

Fit on half A, eval half B (and reverse) — both halves have kgw_pgreen from
the completed train extraction. Compares champion (kgw=entropy) vs kgwpg.
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
d = np.load(C / "kgwpg_train.npz")
assert len(d["kgw_pgreen"]) == len(tr)
tr = tr.copy()
tr["kgw_pgreen"] = d["kgw_pgreen"]

docs_all = tr["document_id"].unique()
ha = set(docs_all[::2])
mask = tr["document_id"].isin(ha).to_numpy()
dfa, dfb = tr[mask].reset_index(drop=True), tr[~mask].reset_index(drop=True)
eva, evb = tre[mask].reset_index(drop=True), tre[~mask].reset_index(drop=True)

BEST = {"unigram": "entropy", "kgw": "entropy", "gumbel": "logprob", "textseal": "logprob"}
logA, logpi = make_transitions(0.03, 80.0)


def direction(fit_df, fit_ev, test_df, tag):
    y = test_df["label"].to_numpy()
    ev_dom = None
    out = {}
    for name, cond, edges in [
        ("champion", BEST, None),
        ("kgwpg [.05,.15,.35,.55]", {**BEST, "kgw": "kgwpg"}, [.05, .15, .35, .55]),
        ("kgwpg [.02,.08,.20,.50,.80]", {**BEST, "kgw": "kgwpg"}, [.02, .08, .20, .50, .80]),
        ("kgwpg [.15,.55]", {**BEST, "kgw": "kgwpg"}, [.15, .55]),
    ]:
        if edges is not None:
            hmm.KGWPG_EDGES = np.array(edges)
        params = fit_emissions(fit_df, fit_ev, use_entropy=cond)
        sc = hmm_scores(test_df, params, logA, logpi)
        t01 = tpr_at_fpr(y, sc, 0.001)
        out[name] = t01
        print(f"  {t01:.4f}   {tag} {name}")
    return out


print("T@0.1%FPR (v2 cache, trans 0.03/80)")
r1 = direction(dfa, eva, dfb, "A->B:")
r2 = direction(dfb, evb, dfa, "B->A:")
print("\nmean deltas vs champion:")
for k in r1:
    if k != "champion":
        d1 = r1[k] - r1["champion"]
        d2 = r2[k] - r2["champion"]
        print(f"  {k}: A->B {d1:+.4f}  B->A {d2:+.4f}  mean {(d1+d2)/2:+.4f}")
