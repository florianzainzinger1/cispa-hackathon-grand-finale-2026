"""5-state HMM over per-token watermark signals -> posterior P(watermarked).

States: 0=Clean, 1=W_unigram, 2=W_kgw, 3=W_gumbel, 4=W_textseal.
Emissions factorize over the four per-token signals (different keys/PRFs are
independent under every state): each signal is binned; per state a categorical
distribution over bins, estimated on train (clean tokens vs spans assigned to
a scheme by max window evidence). Invalid/missing observations (unscored
positions, repeated-ngram dedup) emit uniformly across states (= ignored).

Transitions: Clean<->W_s with geometric priors from label EDA; no direct
W_s->W_s' (true spans are separated by >=12 clean tokens).

Score per token = -log10 P(Clean | all doc signals) via forward-backward.
Boundary bleed cannot happen: emissions are strictly per-token, and the
posterior localizes state switches to within a few tokens.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

SCHEMES = ["unigram", "kgw", "gumbel", "textseal"]

# score bins for continuous signals (null: unit-mean Exp-like, max 9.011)
SCORE_BINS = np.array([0.0, 0.05, 0.15, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.5, 6.5, 9.5])

# conditioning levels for emissions: clean nulls are independent of both
# entropy and token logprob (uniform PRF), but watermark signal strength grows
# with entropy AND falls with token logprob (verified on train: wm gumbel mean
# 1.74 for lp<-3 vs 1.10 for lp>-0.3; clean flat at 1.01).
ENT_EDGES = np.array([0.3, 1.0, 2.0])
LP_EDGES = np.array([-3.0, -1.0, -0.3])
# coarse edges for joint conditioning (keeps bins populated)
ENT_EDGES_C = np.array([1.0])
LP_EDGES_C = np.array([-3.0, -1.0])

N_SIGNAL_BINS = {"unigram": 2, "kgw": 2, "gumbel": len(SCORE_BINS) - 1, "textseal": len(SCORE_BINS) - 1}

# conditioning mode per scheme: None | "entropy" | "logprob" | "both"
DEFAULT_COND = {"unigram": "entropy", "kgw": "entropy", "gumbel": "entropy", "textseal": "entropy"}


# coarse u-score bins used when the argmax-match bit is folded into the code
SCORE_BINS_COARSE = np.array([0.0, 0.15, 0.5, 1.0, 2.0, 4.5, 9.5])


# 1-D edges for the joint dual-key TextSeal observation
DUAL_BINS = np.array([0.0, 0.5, 1.5, 4.0, 9.5])


def signal_bins(scheme: str, mode: str | None) -> int:
    if mode == "lp_match":
        return 2 * (len(SCORE_BINS_COARSE) - 1)  # coarse u-bin x match bit
    if mode == "dual_lp":
        return (len(DUAL_BINS) - 1) ** 2  # joint (u_a, u_b) bins
    return N_SIGNAL_BINS[scheme]


# unigram p_green levels: exact per-position null (mass of the fixed greenlist)
PGREEN_EDGES = np.array([0.35, 0.45, 0.55, 0.65])

# KGW exact per-position greenlist base-mass (cluster top-k extraction);
# null center is gamma=0.25, spread comes from context-dependent greenlists.
# Edges chosen on validation (0.4427 vs champion 0.4384, fit-train protocol).
KGWPG_EDGES = np.array([0.02, 0.10, 0.25, 0.45, 0.65])

# joint kgwpg x entropy conditioning (coarse to keep bins populated)
KGWPG_EDGES_C = np.array([0.02, 0.35])
KGWPG_ENT_EDGES_C = np.array([1.0])


def _n_levels(mode: str | None) -> int:
    if mode is None:
        return 1
    if mode in ("entropy",):
        return len(ENT_EDGES) + 1
    if mode in ("logprob", "lp_match", "dual_lp"):
        return len(LP_EDGES) + 1
    if mode == "pgreen":
        return len(PGREEN_EDGES) + 1
    if mode == "kgwpg":
        return len(KGWPG_EDGES) + 1
    if mode == "kgwpg_ent":
        return (len(KGWPG_EDGES_C) + 1) * (len(KGWPG_ENT_EDGES_C) + 1)
    if mode == "rank":
        return 4  # rank 1 / 2-3 / 4-20 / >20
    if mode == "both":
        return (len(ENT_EDGES_C) + 1) * (len(LP_EDGES_C) + 1)
    raise ValueError(mode)


def _levels(df: pd.DataFrame, mode: str | None) -> np.ndarray:
    if mode is None:
        return np.zeros(len(df), dtype=np.int16)
    ent = np.nan_to_num(df["qwen_entropy"].to_numpy().astype(np.float64), nan=0.5)
    lp = np.nan_to_num(df["qwen_logprob"].to_numpy().astype(np.float64), nan=-1.0)
    if mode == "entropy":
        return np.digitize(ent, ENT_EDGES).astype(np.int16)
    if mode in ("logprob", "lp_match", "dual_lp"):
        return np.digitize(lp, LP_EDGES).astype(np.int16)
    if mode == "pgreen":
        pg = np.nan_to_num(df["unigram_pgreen"].to_numpy().astype(np.float64), nan=0.5)
        return np.digitize(pg, PGREEN_EDGES).astype(np.int16)
    if mode == "kgwpg":
        pg = np.nan_to_num(df["kgw_pgreen"].to_numpy().astype(np.float64), nan=0.25)
        return np.digitize(pg, KGWPG_EDGES).astype(np.int16)
    if mode == "kgwpg_ent":
        pg = np.nan_to_num(df["kgw_pgreen"].to_numpy().astype(np.float64), nan=0.25)
        gi = np.digitize(pg, KGWPG_EDGES_C)
        ei = np.digitize(ent, KGWPG_ENT_EDGES_C)
        return (gi * (len(KGWPG_ENT_EDGES_C) + 1) + ei).astype(np.int16)
    if mode == "rank":
        rk = df["qwen_rank"].to_numpy().astype(np.float64)
        rk = np.where(rk < 1, 2, rk)  # missing (-1) -> mid level
        return np.digitize(rk, [1.5, 3.5, 20.5]).astype(np.int16)
    e = np.digitize(ent, ENT_EDGES_C)
    l = np.digitize(lp, LP_EDGES_C)
    return (e * (len(LP_EDGES_C) + 1) + l).astype(np.int16)


def n_codes(scheme: str, cond) -> int:
    mode = cond.get(scheme) if isinstance(cond, dict) else ("entropy" if cond else None)
    return N_SIGNAL_BINS[scheme] * _n_levels(mode)


def _bin_scores(s: np.ndarray, bins: np.ndarray = SCORE_BINS) -> np.ndarray:
    """Continuous score -> bin index; -1 = missing."""
    out = np.full(len(s), -1, dtype=np.int16)
    ok = ~np.isnan(s)
    out[ok] = np.clip(np.digitize(s[ok], bins) - 1, 0, len(bins) - 2)
    return out


# NP-optimal likelihood-ratio score (deep-research finding, arXiv 2404.01245):
# when set, gumbel/textseal use the nplr_* column as the emission observable
# instead of the Aaronson -log(1-u) score. Bins set from train quantiles.
USE_NPLR = False
NPLR_BINS = None            # np.array of bin edges, shared for gumbel/textseal
_NPLR_COL = {"gumbel": "nplr_gumbel", "textseal": "nplr_ts"}


def set_nplr_bins(train_df: pd.DataFrame, n_bins: int = 12) -> None:
    """Quantile bin edges over pooled valid nplr scores (gumbel+textseal)."""
    global NPLR_BINS, N_SIGNAL_BINS
    vals = []
    for sch, col in _NPLR_COL.items():
        v = train_df[col].to_numpy()
        first = train_df[f"{sch}_first"].to_numpy()
        vals.append(v[(~np.isnan(v)) & (first == 1)])
    pool = np.concatenate(vals)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(pool, qs))
    edges[0], edges[-1] = -np.inf, np.inf
    NPLR_BINS = edges
    N_SIGNAL_BINS = dict(N_SIGNAL_BINS)
    N_SIGNAL_BINS["gumbel"] = len(edges) - 1
    N_SIGNAL_BINS["textseal"] = len(edges) - 1


def _normalize_cond(cond) -> dict:
    """Accept legacy bool (use_entropy) or a per-scheme mode dict."""
    if isinstance(cond, dict):
        return {s: cond.get(s) for s in SCHEMES}
    if cond:
        return dict(DEFAULT_COND)
    return {s: None for s in SCHEMES}


def extract_observations(df: pd.DataFrame, use_entropy=False) -> dict[str, np.ndarray]:
    """Per-token observation codes per signal; -1 = missing/invalid.

    use_entropy: bool (legacy, entropy for all) or per-scheme mode dict.
    Codes live in the product space signal_bin x conditioning_level.
    """
    obs = {}
    uni = df["unigram_green"].to_numpy().astype(np.int16)
    uni_invalid = (df["is_repeated_ngram_2"].to_numpy() == 1) | (
        df["is_special_token"].to_numpy() == 1
    )
    uni = uni.copy()
    uni[uni_invalid] = -1
    obs["unigram"] = uni

    kgw = df["kgw_green"].to_numpy().astype(np.int16).copy()
    kgw_invalid = (kgw < 0) | (df["kgw_first"].to_numpy() == 0)
    kgw[kgw_invalid] = -1
    obs["kgw"] = kgw

    if USE_NPLR and NPLR_BINS is not None:
        g = df[_NPLR_COL["gumbel"]].to_numpy().astype(np.float64).copy()
        g[df["gumbel_first"].to_numpy() == 0] = np.nan
        obs["gumbel"] = _bin_scores(g, NPLR_BINS)
        t = df[_NPLR_COL["textseal"]].to_numpy().astype(np.float64).copy()
        t[df["textseal_first"].to_numpy() == 0] = np.nan
        obs["textseal"] = _bin_scores(t, NPLR_BINS)
    else:
        g = df["gumbel_score"].to_numpy().astype(np.float64).copy()
        g[df["gumbel_first"].to_numpy() == 0] = np.nan
        obs["gumbel"] = _bin_scores(g)
        t = df["textseal_score"].to_numpy().astype(np.float64).copy()
        t[df["textseal_first"].to_numpy() == 0] = np.nan
        obs["textseal"] = _bin_scores(t)

    cond = _normalize_cond(use_entropy)
    for scheme in obs:
        mode = cond[scheme]
        if mode is None:
            continue
        codes = obs[scheme]
        if mode == "dual_lp":
            # joint (u_a, u_b) observation: under wm exactly ONE key is active
            # per token (mixing_alpha) — one score high, the other null. The
            # repo's 50/50 fused score blurs exactly this pattern.
            sa = df["ts_score_a"].to_numpy().astype(np.float64).copy()
            sb = df["ts_score_b"].to_numpy().astype(np.float64).copy()
            invalid = df["textseal_first"].to_numpy() == 0
            sa[invalid] = np.nan
            nb1 = len(DUAL_BINS) - 1
            codes = np.full(len(df), -1, dtype=np.int16)
            ok = ~np.isnan(sa) & ~np.isnan(sb)
            ba = np.clip(np.digitize(sa[ok], DUAL_BINS) - 1, 0, nb1 - 1)
            bb = np.clip(np.digitize(sb[ok], DUAL_BINS) - 1, 0, nb1 - 1)
            codes[ok] = ba + nb1 * bb
            obs[scheme] = codes
        if mode == "lp_match":
            # rebuild base code: coarse u-bin x argmax-match bit
            if scheme == "gumbel":
                score = df["gumbel_score"].to_numpy().astype(np.float64).copy()
                score[df["gumbel_first"].to_numpy() == 0] = np.nan
                match = df["gumbel_match"].to_numpy().astype(np.int16)
            else:
                score = df["textseal_score"].to_numpy().astype(np.float64).copy()
                score[df["textseal_first"].to_numpy() == 0] = np.nan
                match = np.maximum(
                    df["ts_match_a"].to_numpy(), df["ts_match_b"].to_numpy()
                ).astype(np.int16)
            nb_u = len(SCORE_BINS_COARSE) - 1
            codes = np.full(len(df), -1, dtype=np.int16)
            ok = ~np.isnan(score) & (match >= 0)
            ubin = np.clip(np.digitize(score[ok], SCORE_BINS_COARSE) - 1, 0, nb_u - 1)
            codes[ok] = ubin + nb_u * match[ok]
            obs[scheme] = codes
        level = _levels(df, mode)
        ok = obs[scheme] >= 0
        obs[scheme][ok] = obs[scheme][ok] + signal_bins(scheme, mode) * level[ok]
    return obs


def assign_spans(df: pd.DataFrame, ev: pd.DataFrame) -> np.ndarray:
    """Per-token scheme assignment (0..3) for label=1 tokens, -1 elsewhere.

    Each contiguous label-1 span is assigned to the scheme with the highest
    mean window evidence inside the span.
    """
    assign = np.full(len(df), -1, dtype=np.int8)
    y = df["label"].to_numpy()
    docs = df["document_id"].to_numpy()
    ev_mat = ev[SCHEMES].to_numpy()
    start = None
    for i in range(len(df) + 1):
        boundary = i == len(df) or y[i] == 0 or (i > 0 and docs[i] != docs[i - 1])
        if start is not None and boundary:
            scheme = int(ev_mat[start:i].mean(axis=0).argmax())
            assign[start:i] = scheme
            start = None
        if i < len(df) and y[i] == 1 and start is None and not boundary:
            start = i
        elif i < len(df) and y[i] == 1 and start is None:
            start = i
    return assign


# logprob channel: watermarked text is sampled at temp 0.9 with bias; human
# and clean-generated text have different actual-token logprob profiles.
LOGPROB_BINS = np.array([-np.inf, -6, -4, -3, -2.2, -1.6, -1.1, -0.7, -0.4, -0.2, -0.05, 0.001])


def _logprob_codes(df: pd.DataFrame) -> np.ndarray:
    lp = df["qwen_logprob"].to_numpy().astype(np.float64)
    out = np.full(len(lp), -1, dtype=np.int16)
    ok = ~np.isnan(lp)
    out[ok] = np.clip(np.digitize(lp[ok], LOGPROB_BINS) - 1, 0, len(LOGPROB_BINS) - 2)
    return out


def fit_emissions(df: pd.DataFrame, ev: pd.DataFrame, eps: float = 0.5, use_entropy: bool = False) -> dict:
    """Per-signal categorical emissions for Clean + the signal's own W state.

    With use_entropy: conditional distributions P(bin | ent_level, state) —
    normalized WITHIN each entropy level so that the (state-independent)
    entropy of the text itself does not leak into the likelihood ratio.
    """
    obs = extract_observations(df, use_entropy)
    assign = assign_spans(df, ev)
    y = df["label"].to_numpy()
    cond = _normalize_cond(use_entropy)

    params: dict = {"use_entropy": use_entropy}
    for si, scheme in enumerate(SCHEMES):
        codes = obs[scheme]
        nb = signal_bins(scheme, cond[scheme])
        n_levels = _n_levels(cond[scheme])
        total = nb * n_levels
        clean_sel = (y == 0) & (codes >= 0)
        wm_sel = (assign == si) & (codes >= 0)
        p_clean = (np.bincount(codes[clean_sel], minlength=total) + eps).astype(np.float64)
        p_wm = (np.bincount(codes[wm_sel], minlength=total) + eps).astype(np.float64)
        # normalize per conditioning level (conditional distributions: the
        # state-independent level itself must not leak into the likelihood)
        for lv in range(n_levels):
            sl = slice(lv * nb, (lv + 1) * nb)
            p_clean[sl] /= p_clean[sl].sum()
            p_wm[sl] /= p_wm[sl].sum()
        params[scheme] = {
            "log_clean": np.log(p_clean),
            "log_wm": np.log(p_wm),
            "n_wm_tokens": int(wm_sel.sum()),
        }
    return params


def emission_loglik(df: pd.DataFrame, params: dict) -> np.ndarray:
    """(n_tokens, 5) log-likelihood matrix. Missing obs contribute 0 to all."""
    obs = extract_observations(df, params.get("use_entropy", False))
    n = len(df)
    ll = np.zeros((n, 5), dtype=np.float64)
    for si, scheme in enumerate(SCHEMES):
        codes = obs[scheme]
        ok = codes >= 0
        lc = params[scheme]["log_clean"][codes[ok]]
        lw = params[scheme]["log_wm"][codes[ok]]
        # clean-state and all foreign W states see the null distribution
        for state in range(5):
            ll[ok, state] += np.where(state == si + 1, lw, lc)
    return ll


def make_transitions(
    p_enter: float = 0.004,   # Clean -> any W (spans per clean token)
    mean_span: float = 110.0,  # geometric mean span length
    p_start_wm: float = 0.07,  # fraction of docs starting inside a span
) -> tuple[np.ndarray, np.ndarray]:
    A = np.zeros((5, 5))
    A[0, 0] = 1.0 - p_enter
    for s in range(1, 5):
        A[0, s] = p_enter / 4
        A[s, s] = 1.0 - 1.0 / mean_span
        A[s, 0] = 1.0 / mean_span
    pi = np.array([1.0 - p_start_wm] + [p_start_wm / 4] * 4)
    return np.log(A + 1e-300), np.log(pi)


def forward_backward(ll: np.ndarray, logA: np.ndarray, logpi: np.ndarray) -> np.ndarray:
    """Posterior P(state|obs) for one document. ll: (T,5)."""
    T, S = ll.shape
    alpha = np.zeros((T, S))
    beta = np.zeros((T, S))
    alpha[0] = logpi + ll[0]
    for t in range(1, T):
        m = alpha[t - 1][:, None] + logA
        alpha[t] = ll[t] + np.logaddexp.reduce(m, axis=0)
    beta[T - 1] = 0.0
    for t in range(T - 2, -1, -1):
        m = logA + (ll[t + 1] + beta[t + 1])[None, :]
        beta[t] = np.logaddexp.reduce(m, axis=1)
    post = alpha + beta
    post -= np.logaddexp.reduce(post, axis=1, keepdims=True)
    return np.exp(post)


def hmm_scores(df: pd.DataFrame, params: dict, logA: np.ndarray, logpi: np.ndarray) -> np.ndarray:
    """-log10 P(Clean | signals) per token, all documents."""
    ll = emission_loglik(df, params)
    docs = df["document_id"].to_numpy()
    out = np.zeros(len(df), dtype=np.float64)
    start = 0
    for i in range(1, len(df) + 1):
        if i == len(df) or docs[i] != docs[start]:
            post = forward_backward(ll[start:i], logA, logpi)
            p_clean = np.clip(post[:, 0], 1e-30, 1.0)
            out[start:i] = -np.log10(p_clean)
            start = i
    return out
