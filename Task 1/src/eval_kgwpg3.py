"""kgwpg robustness: split-half + reverse-split (challenger-round protocol).

Candidate must beat champion on: full val (done: 0.4427/0.4452), BOTH val
halves (fit train), and BOTH reverse splits (fit train+otherhalf). Decides
between edge sets E5=[.02,.10,.25,.45,.65] and E7=[.01,.04,.10,.20,.35,.55,.80].
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hmm  # noqa
from hmm import fit_emissions, make_transitions, hmm_scores  # noqa
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

docs = pd.unique(va["document_id"])
half_a = set(docs[::2])
in_a = va["document_id"].isin(half_a).to_numpy()
va_a, vae_a = va[in_a], vae[in_a]
va_b, vae_b = va[~in_a], vae[~in_a]

BEST = {"unigram": "entropy", "kgw": "entropy", "gumbel": "logprob", "textseal": "logprob"}
KP = {**BEST, "kgw": "kgwpg"}
E5 = [.02, .10, .25, .45, .65]
E7 = [.01, .04, .10, .20, .35, .55, .80]

logA, logpi = make_transitions(0.03, 80.0)

def score(fit_df, fit_ev, eval_df, cond, edges=None):
    if edges is not None:
        hmm.KGWPG_EDGES = np.array(edges)
    params = fit_emissions(fit_df, fit_ev, use_entropy=cond)
    sc = hmm_scores(eval_df, params, logA, logpi)
    return tpr_at_fpr(eval_df["label"].to_numpy(), sc, 0.001)

configs = [("champion", BEST, None), ("kgwpg E5", KP, E5), ("kgwpg E7", KP, E7)]

print("split-half (fit train):")
for tag, cond, e in configs:
    a = score(tr, tre, va_a, cond, e)
    b = score(tr, tre, va_b, cond, e)
    print(f"  {tag:10s} halfA {a:.4f}  halfB {b:.4f}")

print("reverse-split (fit train+other half):")
tr_a = pd.concat([tr, va_a], ignore_index=True)
tre_a = pd.concat([tre, vae_a], ignore_index=True)
tr_b = pd.concat([tr, va_b], ignore_index=True)
tre_b = pd.concat([tre, vae_b], ignore_index=True)
for tag, cond, e in configs:
    a = score(tr_b, tre_b, va_a, cond, e)  # eval A, fit train+B
    b = score(tr_a, tre_a, va_b, cond, e)  # eval B, fit train+A
    print(f"  {tag:10s} evalA {a:.4f}  evalB {b:.4f}")
