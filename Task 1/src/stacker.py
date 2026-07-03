"""Gradient-boosting stacker over HMM posterior + all cached features.

Evaluation protocol (leak-free):
  - HMM posterior features are CROSS-FITTED: 5 doc-level folds over
    train+validation; fold k's posterior uses emissions fitted on the other
    folds only.
  - GBM evaluated via GroupKFold by document_id on those out-of-fold
    posterior features; reported metric = pooled tpr_at_fpr over all OOF
    predictions. Baseline to beat = the cross-fitted HMM posterior itself
    under the same pooling.

Test-time: HMM emissions fit on all labeled docs, GBM fit on all labeled rows.

Usage:
  python src/stacker.py                 # CV evaluation only
  python src/stacker.py --out submissions/sub_007_stacker.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hmm import fit_emissions, make_transitions, hmm_scores  # noqa: E402
from metrics import tpr_at_fpr  # noqa: E402

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
COND_BEST = {"unigram": "entropy", "kgw": "entropy", "gumbel": "logprob", "textseal": "logprob"}
P_ENTER, MEAN_SPAN = 0.02, 45.0
N_FOLDS = 5
SQUASH = 8.0

EV_COLS = ["unigram", "kgw", "gumbel", "textseal", "fused"]
RAW_COLS = [
    "unigram_green", "kgw_green", "gumbel_score", "textseal_score",
    "qwen_entropy", "qwen_logprob",
    "is_special_token", "is_repeated_ngram_2", "is_repeated_ngram_3",
    "is_number_like", "is_newline_like", "token_piece_length",
]


def load_labeled() -> tuple[pd.DataFrame, pd.DataFrame]:
    dfs, evs = [], []
    for split in ("train", "validation"):
        dfs.append(pd.read_parquet(CACHE_DIR / f"{split}_signals.parquet"))
        evs.append(pd.read_parquet(CACHE_DIR / f"{split}_evidence.parquet"))
    return pd.concat(dfs, ignore_index=True), pd.concat(evs, ignore_index=True)


def crossfit_hmm(df: pd.DataFrame, ev: pd.DataFrame) -> np.ndarray:
    """Out-of-fold HMM posterior for every labeled token."""
    logA, logpi = make_transitions(P_ENTER, MEAN_SPAN)
    docs = df["document_id"].to_numpy()
    uniq = pd.unique(docs)
    rng = np.random.default_rng(7)
    order = rng.permutation(len(uniq))
    fold_of_doc = {d: order[i] % N_FOLDS for i, d in enumerate(uniq)}
    fold = np.array([fold_of_doc[d] for d in docs])

    out = np.zeros(len(df))
    for k in range(N_FOLDS):
        tr_sel = fold != k
        params = fit_emissions(
            df[tr_sel].reset_index(drop=True),
            ev[tr_sel].reset_index(drop=True),
            use_entropy=COND_BEST,
        )
        sub = df[~tr_sel].reset_index(drop=True)
        out[~tr_sel] = hmm_scores(sub, params, logA, logpi)
    return out, fold


def feature_matrix(df: pd.DataFrame, ev: pd.DataFrame, hmm_post: np.ndarray) -> np.ndarray:
    n = len(df)
    pos = df["position"].to_numpy().astype(np.float32)
    doc_len = df.groupby("document_id")["position"].transform("max").to_numpy() + 1
    feats = [hmm_post.astype(np.float32), pos, (pos / doc_len).astype(np.float32)]
    for c in EV_COLS:
        feats.append(ev[c].to_numpy().astype(np.float32))
    for c in RAW_COLS:
        v = df[c].to_numpy().astype(np.float32)
        feats.append(np.nan_to_num(v, nan=-1.0))
    rank = np.clip(df["qwen_rank"].to_numpy().astype(np.float32), -1, 1000)
    feats.append(np.log1p(np.maximum(rank, 0)))
    return np.stack(feats, axis=1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    df, ev = load_labeled()
    y = df["label"].to_numpy()
    print("cross-fitting HMM posterior (5 folds)...")
    hmm_oof, fold = crossfit_hmm(df, ev)
    base = tpr_at_fpr(y, hmm_oof)
    print(f"BASELINE cross-fitted HMM posterior: OOF pooled @0.1%={base:.4f} @1%={tpr_at_fpr(y, hmm_oof, 0.01):.4f}")

    X = feature_matrix(df, ev, hmm_oof)
    oof_pred = np.zeros(len(df))
    for k in range(N_FOLDS):
        tr_sel = fold != k
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
            min_samples_leaf=200, l2_regularization=1.0, random_state=0,
        )
        clf.fit(X[tr_sel], y[tr_sel])
        oof_pred[~tr_sel] = clf.predict_proba(X[~tr_sel])[:, 1]
    gbm = tpr_at_fpr(y, oof_pred)
    print(f"STACKER GBM: OOF pooled @0.1%={gbm:.4f} @1%={tpr_at_fpr(y, oof_pred, 0.01):.4f}")

    # rank-average blend
    from scipy.stats import rankdata
    blend = rankdata(hmm_oof) + rankdata(oof_pred)
    bl = tpr_at_fpr(y, blend)
    print(f"BLEND (rank avg): OOF pooled @0.1%={bl:.4f} @1%={tpr_at_fpr(y, blend, 0.01):.4f}")

    if args.out is None:
        return 0

    choice = max([(base, "hmm"), (gbm, "gbm"), (bl, "blend")])
    print(f"winner: {choice[1]} ({choice[0]:.4f})")
    if choice[1] == "hmm":
        print("stacker does not beat HMM — not writing a submission")
        return 0

    # full fit for test
    te = pd.read_parquet(CACHE_DIR / "test_signals.parquet")
    te_ev = pd.read_parquet(CACHE_DIR / "test_evidence.parquet")
    logA, logpi = make_transitions(P_ENTER, MEAN_SPAN)
    params = fit_emissions(df, ev, use_entropy=COND_BEST)
    te_hmm = hmm_scores(te, params, logA, logpi)
    X_te = feature_matrix(te, te_ev, te_hmm)
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
        min_samples_leaf=200, l2_regularization=1.0, random_state=0,
    )
    clf.fit(X, y)
    p_te = clf.predict_proba(X_te)[:, 1]

    if choice[1] == "gbm":
        final = p_te
    else:
        final = (rankdata(te_hmm) + rankdata(p_te)) / (2 * len(te_hmm))

    final = np.clip(final, 0.0, 1.0)
    docs = te["document_id"].to_numpy()
    args.out.parent.mkdir(exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for doc_id in pd.unique(docs):
            s = final[docs == doc_id]
            fh.write(json.dumps(
                {"document_id": str(doc_id), "scores": [round(float(x), 6) for x in s]},
                separators=(",", ":"),
            ) + "\n")
    print(f"submission written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
