"""kgwpg with per-token LR cap — kills the heavy-tail FP mechanism that made
raw kgwpg fail the challenger gate (report 14): rare chance-greens at thin
low-pgreen levels got huge LRs and ate the per-half 0.1% FPR budget.

Cap: clip the kgw-channel log-LR (log_wm - log_clean) at +/- log(C).
Gate protocol = eval_kgwpg3: pooled val + BOTH val halves + BOTH reverse
splits, each must beat champion. Ship only on 5/5.
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
logA, logpi = make_transitions(0.03, 80.0)


def cap_kgw(params, cap):
    """Clip kgw-channel log-LR to +/- log(cap)."""
    if cap is None:
        return params
    p = dict(params)
    kg = dict(p["kgw"])
    llr = np.clip(kg["log_wm"] - kg["log_clean"], -np.log(cap), np.log(cap))
    kg["log_wm"] = kg["log_clean"] + llr
    p["kgw"] = kg
    return p


def score(fit_df, fit_ev, eval_df, cond, edges=None, cap=None):
    if edges is not None:
        hmm.KGWPG_EDGES = np.array(edges)
    params = cap_kgw(fit_emissions(fit_df, fit_ev, use_entropy=cond), cap)
    sc = hmm_scores(eval_df, params, logA, logpi)
    return tpr_at_fpr(eval_df["label"].to_numpy(), sc, 0.001)


tr_a = pd.concat([tr, va_a], ignore_index=True)
tre_a = pd.concat([tre, vae_a], ignore_index=True)
tr_b = pd.concat([tr, va_b], ignore_index=True)
tre_b = pd.concat([tre, vae_b], ignore_index=True)


def gate(tag, cond, edges=None, cap=None):
    full = score(tr, tre, va, cond, edges, cap)
    ha = score(tr, tre, va_a, cond, edges, cap)
    hb = score(tr, tre, va_b, cond, edges, cap)
    ra = score(tr_b, tre_b, va_a, cond, edges, cap)
    rb = score(tr_a, tre_a, va_b, cond, edges, cap)
    print(f"  {tag:22s} val {full:.4f}  halfA {ha:.4f}  halfB {hb:.4f}  revA {ra:.4f}  revB {rb:.4f}")
    return np.array([full, ha, hb, ra, rb])


print("challenger gate (5 numbers each; ship needs 5/5 vs champion):")
ch = gate("champion", BEST)
for cap in (16.0, 8.0, 4.0, 2.5):
    r = gate(f"kgwpg E5 cap={cap}", KP, E5, cap)
    wins = (r > ch).sum()
    print(f"    -> beats champion on {wins}/5")
