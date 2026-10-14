"""Evaluate the NP-optimal likelihood-ratio score vs the Aaronson score.

Merges nplr_{split}.npz (cols nplr_gumbel, nplr_tsa, nplr_tsb) into the cached
signals, derives nplr_ts = marginal dual-key LR (logmeanexp of the two keys),
then compares HMM validation TPR@0.1%FPR with USE_NPLR off vs on.

Usage: python src/test_nplr.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hmm as H  # noqa: E402
from metrics import tpr_at_fpr  # noqa: E402
from load_data import load_split  # noqa: E402

CACHE = Path(__file__).resolve().parent.parent / "cache"


def merge_nplr(split: str) -> pd.DataFrame:
    df = pd.read_parquet(CACHE / f"{split}_signals.parquet")
    d = np.load(CACHE / f"nplr_{split}.npz")
    rows = load_split(split)
    lens = d["lengths"]
    assert lens.sum() == len(df), (lens.sum(), len(df))
    assert (np.repeat([r["document_id"] for r in rows], lens) == df["document_id"].to_numpy()).all()
    df["nplr_gumbel"] = d["nplr_gumbel"]
    a, b = d["nplr_tsa"], d["nplr_tsb"]
    # marginal dual-key LR with equal prior: log( (e^a + e^b)/2 )
    df["nplr_ts"] = np.logaddexp(a, b) - np.log(2.0)
    return df


def main() -> int:
    tr = merge_nplr("train")
    va = merge_nplr("validation")
    tr_ev = pd.read_parquet(CACHE / "train_evidence.parquet")
    yva = va["label"].to_numpy()
    ytr = tr["label"].to_numpy()
    logA, logpi = H.make_transitions(0.02, 45)

    # sanity: does nplr separate labels better than the Aaronson score?
    for col in ("gumbel_score", "nplr_gumbel", "textseal_score", "nplr_ts"):
        s = tr[col].to_numpy(); ok = ~np.isnan(s)
        print(f"train {col:14s}: mean|0={s[ok&(ytr==0)].mean():.4f} mean|1={s[ok&(ytr==1)].mean():.4f}")

    cond = {"unigram": "entropy", "kgw": "entropy", "gumbel": "logprob", "textseal": "logprob"}

    # baseline (Aaronson score)
    H.USE_NPLR = False
    p0 = H.fit_emissions(tr, tr_ev, use_entropy=cond)
    s0 = H.hmm_scores(va, p0, logA, logpi)
    print(f"\nbaseline (Aaronson -log(1-u)): val @0.1%={tpr_at_fpr(yva,s0):.4f} @1%={tpr_at_fpr(yva,s0,0.01):.4f}")

    # NP-LR score
    for nb in (10, 14, 20):
        H.USE_NPLR = True
        H.set_nplr_bins(tr, n_bins=nb)
        p1 = H.fit_emissions(tr, tr_ev, use_entropy=cond)
        s1 = H.hmm_scores(va, p1, logA, logpi)
        print(f"NP-LR score ({nb} bins):        val @0.1%={tpr_at_fpr(yva,s1):.4f} @1%={tpr_at_fpr(yva,s1,0.01):.4f}")

    # split-half robustness with best bin count (14)
    H.USE_NPLR = True; H.set_nplr_bins(tr, 14)
    docs = tr["document_id"].unique()
    selA = tr["document_id"].isin(set(docs[:45])).to_numpy()
    trA, trB = tr[selA].reset_index(drop=True), tr[~selA].reset_index(drop=True)
    evA, evB = tr_ev[selA].reset_index(drop=True), tr_ev[~selA].reset_index(drop=True)
    for tag, fit_d, fit_e, ev_d in (("A->B", trA, evA, trB), ("B->A", trB, evB, trA)):
        H.set_nplr_bins(fit_d, 14)
        p = H.fit_emissions(fit_d, fit_e, use_entropy=cond)
        sc = H.hmm_scores(ev_d, p, logA, logpi)
        yb = ev_d["label"].to_numpy()
        print(f"NP-LR split {tag}: @0.1%={tpr_at_fpr(yb,sc):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
