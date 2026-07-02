"""Challenger A: hard-negative tail veto for the champion HMM.

Idea: the pooled TPR@0.1%FPR metric is decided entirely by the few hundred
highest-scoring CLEAN tokens (the false-positive tail). Train a classifier
only on the critical score zone (champion score above the split's clean
99.5th percentile) to recognize clean tokens that sneaked into the tail, and
subtract alpha * P(clean|features) from the champion score inside the zone.

Honesty protocol:
  - champion scores are computed cross-split (emissions fit on the OTHER
    split), so every labeled token has an out-of-sample champion score;
  - veto probabilities are out-of-fold (GroupKFold(5) by document_id over
    the pooled zone tokens of both splits);
  - a stricter cross-split veto (model fit only on the other split's zone)
    and a split-half check within train (emissions AND veto fit on 45 docs,
    evaluated on the other 45) are also reported.

Local evaluation only -- this script never writes a submission.

Usage:  python src/tail_veto.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hmm import fit_emissions, hmm_scores, make_transitions  # noqa: E402
from metrics import tpr_at_fpr  # noqa: E402

CACHE = Path(__file__).resolve().parent.parent / "cache"

# champion configuration (validation TPR@0.1%FPR = 0.4255)
COND = {"unigram": "entropy", "kgw": "entropy", "gumbel": "logprob", "textseal": "logprob"}
P_ENTER, MEAN_SPAN = 0.02, 45.0

ALPHAS = [0.5, 1.0, 2.0, 4.0]
ZONE_PCT = 99.5   # zone = champion score above this clean percentile
PEAK_PCT = 99.9   # "peak" tokens for the distance feature
RANK_CLIP = 100_000.0

EV_COLS = ["unigram", "kgw", "gumbel", "textseal", "fused"]
FLAG_COLS = [
    "is_special_token",
    "is_repeated_ngram_2",
    "is_repeated_ngram_3",
    "is_number_like",
    "is_newline_like",
]
FEATURE_NAMES = (
    ["champ_score"]
    + [f"ev_{c}" for c in EV_COLS]
    + ["qwen_entropy", "qwen_logprob", "log1p_rank", "token_piece_length"]
    + FLAG_COLS
    + ["position", "doc_length", "rel_position", "dist_to_peak"]
)


def load_split(split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(CACHE / f"{split}_signals.parquet")
    ev = pd.read_parquet(CACHE / f"{split}_evidence.parquet")
    assert len(df) == len(ev)
    return df, ev


def champion_scores(df_fit, ev_fit, df_score) -> np.ndarray:
    params = fit_emissions(df_fit, ev_fit, use_entropy=COND)
    logA, logpi = make_transitions(P_ENTER, MEAN_SPAN)
    return hmm_scores(df_score, params, logA, logpi)


def doc_blocks(docs: np.ndarray):
    """Yield (start, end) of contiguous same-document row blocks."""
    start = 0
    for i in range(1, len(docs) + 1):
        if i == len(docs) or docs[i] != docs[start]:
            yield start, i
            start = i


def build_features(df: pd.DataFrame, ev: pd.DataFrame, champ: np.ndarray,
                   peak_thr: float) -> np.ndarray:
    """Feature matrix (n_tokens, n_features), NaN-friendly for HGB."""
    n = len(df)
    docs = df["document_id"].to_numpy()

    doc_len = np.zeros(n)
    dist_peak = np.zeros(n)
    for s, e in doc_blocks(docs):
        T = e - s
        doc_len[s:e] = T
        idx = np.arange(T)
        peaks = idx[champ[s:e] > peak_thr]
        if len(peaks) == 0:
            dist_peak[s:e] = T
        else:
            j = np.clip(np.searchsorted(peaks, idx), 0, len(peaks) - 1)
            d_right = np.abs(peaks[j] - idx)
            d_left = np.abs(idx - peaks[np.maximum(j - 1, 0)])
            dist_peak[s:e] = np.minimum(d_right, d_left)

    rank = df["qwen_rank"].to_numpy().astype(np.float64)
    log_rank = np.log1p(np.clip(rank, 0.0, RANK_CLIP))
    position = df["position"].to_numpy().astype(np.float64)

    cols = [champ]
    cols += [ev[c].to_numpy().astype(np.float64) for c in EV_COLS]
    cols += [
        df["qwen_entropy"].to_numpy().astype(np.float64),
        df["qwen_logprob"].to_numpy().astype(np.float64),
        log_rank,
        df["token_piece_length"].to_numpy().astype(np.float64),
    ]
    cols += [df[c].to_numpy().astype(np.float64) for c in FLAG_COLS]
    cols += [position, doc_len, position / doc_len, dist_peak]
    X = np.column_stack(cols)
    assert X.shape[1] == len(FEATURE_NAMES)
    return X


def make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=0,
    )


def oof_p_clean(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Out-of-fold P(clean|features) via GroupKFold(5) by document."""
    p_clean = np.full(len(y), float((y == 0).mean()))
    gkf = GroupKFold(n_splits=5)
    for tr_idx, te_idx in gkf.split(X, y, groups):
        if len(np.unique(y[tr_idx])) < 2:
            continue  # degenerate fold: keep prior
        m = make_model().fit(X[tr_idx], y[tr_idx])
        p_clean[te_idx] = m.predict_proba(X[te_idx])[:, 0]
    return p_clean


def veto_eval(y, champ, zone, p_clean_zone, alpha) -> float:
    final = champ.copy()
    final[zone] -= alpha * p_clean_zone
    return tpr_at_fpr(y, final)


def main() -> int:
    rows = []
    tr_df, tr_ev = load_split("train")
    va_df, va_ev = load_split("validation")
    y_tr = tr_df["label"].to_numpy()
    y_va = va_df["label"].to_numpy()

    # ---- honest cross-split champion scores ----------------------------
    print("computing champion scores (train-emissions -> validation) ...")
    sc_va = champion_scores(tr_df, tr_ev, va_df)
    print("computing champion scores (validation-emissions -> train) ...")
    sc_tr = champion_scores(va_df, va_ev, tr_df)

    base_va = tpr_at_fpr(y_va, sc_va)
    base_tr = tpr_at_fpr(y_tr, sc_tr)
    print(f"\nbaseline champion  validation TPR@0.1%FPR = {base_va:.4f}"
          f"   train (val-fit) = {base_tr:.4f}")

    # ---- critical zone per split ---------------------------------------
    thr_va = np.percentile(sc_va[y_va == 0], ZONE_PCT)
    thr_tr = np.percentile(sc_tr[y_tr == 0], ZONE_PCT)
    peak_va = np.percentile(sc_va[y_va == 0], PEAK_PCT)
    peak_tr = np.percentile(sc_tr[y_tr == 0], PEAK_PCT)
    zone_va = sc_va > thr_va
    zone_tr = sc_tr > thr_tr
    print(f"zone thresholds: val clean-p{ZONE_PCT}={thr_va:.3f} "
          f"(zone {zone_va.sum()} tok, {int((zone_va & (y_va == 0)).sum())} clean) | "
          f"train clean-p{ZONE_PCT}={thr_tr:.3f} "
          f"(zone {zone_tr.sum()} tok, {int((zone_tr & (y_tr == 0)).sum())} clean)")

    # ---- features & pooled OOF veto ------------------------------------
    X_tr = build_features(tr_df, tr_ev, sc_tr, peak_tr)
    X_va = build_features(va_df, va_ev, sc_va, peak_va)

    Xz = np.vstack([X_tr[zone_tr], X_va[zone_va]])
    yz = np.concatenate([y_tr[zone_tr], y_va[zone_va]])
    gz = np.concatenate([
        "tr::" + tr_df["document_id"].to_numpy()[zone_tr],
        "va::" + va_df["document_id"].to_numpy()[zone_va],
    ])
    print(f"\npooled zone: {len(yz)} tokens, {(yz == 0).sum()} clean; OOF GroupKFold(5) ...")
    pc = oof_p_clean(Xz, yz, gz)
    n_ztr = int(zone_tr.sum())
    pc_tr, pc_va = pc[:n_ztr], pc[n_ztr:]

    # stricter variant: veto model fit purely on the OTHER split's zone
    m_tr = make_model().fit(X_tr[zone_tr], y_tr[zone_tr])
    m_va = make_model().fit(X_va[zone_va], y_va[zone_va])
    pc_va_x = m_tr.predict_proba(X_va[zone_va])[:, 0]
    pc_tr_x = m_va.predict_proba(X_tr[zone_tr])[:, 0]

    print(f"\n{'alpha':>6} | {'val OOF':>8} {'val Xsplit':>10} | "
          f"{'train OOF':>9} {'train Xsplit':>12}")
    print(f"{'base':>6} | {base_va:8.4f} {base_va:10.4f} | {base_tr:9.4f} {base_tr:12.4f}")
    for a in ALPHAS:
        v1 = veto_eval(y_va, sc_va, zone_va, pc_va, a)
        v2 = veto_eval(y_va, sc_va, zone_va, pc_va_x, a)
        t1 = veto_eval(y_tr, sc_tr, zone_tr, pc_tr, a)
        t2 = veto_eval(y_tr, sc_tr, zone_tr, pc_tr_x, a)
        rows.append((a, v1, v2, t1, t2))
        print(f"{a:6.1f} | {v1:8.4f} {v2:10.4f} | {t1:9.4f} {t2:12.4f}")

    # ---- split-half check within train ----------------------------------
    print("\nsplit-half check (fit emissions AND veto on 45 train docs, eval other 45):")
    uniq = np.sort(tr_df["document_id"].unique())
    half_a = set(uniq[::2])
    mask_a = tr_df["document_id"].isin(half_a).to_numpy()
    halves = {}
    for name, m in (("A", mask_a), ("B", ~mask_a)):
        halves[name] = (
            tr_df.loc[m].reset_index(drop=True),
            tr_ev.loc[m].reset_index(drop=True),
        )
    sh_results = {}
    for fit_h, ev_h in (("A", "B"), ("B", "A")):
        f_df, f_ev = halves[fit_h]
        e_df, e_ev = halves[ev_h]
        params = fit_emissions(f_df, f_ev, use_entropy=COND)
        logA, logpi = make_transitions(P_ENTER, MEAN_SPAN)
        sc_fit = hmm_scores(f_df, params, logA, logpi)     # in-sample (veto training)
        sc_ev = hmm_scores(e_df, params, logA, logpi)      # honest eval scores
        yf = f_df["label"].to_numpy()
        ye = e_df["label"].to_numpy()
        thr_f = np.percentile(sc_fit[yf == 0], ZONE_PCT)
        pk_f = np.percentile(sc_fit[yf == 0], PEAK_PCT)
        thr_e = np.percentile(sc_ev[ye == 0], ZONE_PCT)
        pk_e = np.percentile(sc_ev[ye == 0], PEAK_PCT)
        zf = sc_fit > thr_f
        ze = sc_ev > thr_e
        Xf = build_features(f_df, f_ev, sc_fit, pk_f)
        Xe = build_features(e_df, e_ev, sc_ev, pk_e)
        base = tpr_at_fpr(ye, sc_ev)
        if len(np.unique(yf[zf])) < 2:
            print(f"  fit {fit_h} -> eval {ev_h}: degenerate zone, skipped")
            continue
        m = make_model().fit(Xf[zf], yf[zf])
        pce = m.predict_proba(Xe[ze])[:, 0]
        res = [veto_eval(ye, sc_ev, ze, pce, a) for a in ALPHAS]
        sh_results[(fit_h, ev_h)] = (base, res)
        line = "  ".join(f"a={a:g}:{r:.4f}" for a, r in zip(ALPHAS, res))
        print(f"  fit {fit_h} -> eval {ev_h}: base={base:.4f}  {line}")

    # ---- top-10 vetoed clean tokens on validation (OOF probabilities) ---
    clean_zone = zone_va & (y_va == 0)
    zidx = np.flatnonzero(zone_va)
    df_zone = pd.DataFrame(X_va[zone_va], columns=FEATURE_NAMES)
    df_zone.insert(0, "document_id", va_df["document_id"].to_numpy()[zone_va])
    df_zone["position"] = va_df["position"].to_numpy()[zone_va]
    df_zone["p_clean_oof"] = pc_va
    df_zone["label"] = y_va[zone_va]
    top = (df_zone[df_zone["label"] == 0]
           .sort_values("p_clean_oof", ascending=False).head(10))
    show = ["document_id", "position", "champ_score", "p_clean_oof", "ev_fused",
            "qwen_entropy", "qwen_logprob", "log1p_rank", "token_piece_length",
            "is_repeated_ngram_2", "is_number_like", "is_newline_like",
            "dist_to_peak", "rel_position"]
    print("\ntop-10 vetoed CLEAN validation tokens (by OOF P(clean)):")
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(top[show].to_string(index=False))

    # unused-variable guard for linters
    _ = zidx
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
