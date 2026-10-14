"""Window features: green-indicator -> exact binomial -log10(p) per token.

Under the clean null, the green count in a window of n tokens is
Binomial(n, gamma). Evidence per token = tail p-value of its window counts,
reported as -log10(p) (higher = more watermark-like). Windows never cross
document boundaries; window shapes: centered / left-only / right-only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binom

WINDOW_SIZES = (16, 32, 64, 128)
SHAPES = ("center", "left", "right")


def _window_counts(x: np.ndarray, w: int, shape: str) -> tuple[np.ndarray, np.ndarray]:
    """Green count + actual window length for every position (one document)."""
    n = len(x)
    acc_dtype = np.float64 if np.issubdtype(x.dtype, np.floating) else np.int64
    cs = np.concatenate([np.zeros(1, dtype=acc_dtype), np.cumsum(x, dtype=acc_dtype)])
    idx = np.arange(n)
    if shape == "left":
        lo, hi = np.maximum(0, idx - w + 1), idx + 1
    elif shape == "right":
        lo, hi = idx, np.minimum(n, idx + w)
    else:  # centered
        half = w // 2
        lo, hi = np.maximum(0, idx - half), np.minimum(n, idx - half + w)
    return cs[hi] - cs[lo], (hi - lo)


def _neglog10_binom_sf(k: np.ndarray, n: np.ndarray, gamma: float) -> np.ndarray:
    """-log10 P(X >= k), X ~ Binomial(n, gamma). Uses logsf for tail precision."""
    # sf(k-1) = P(X >= k)
    logsf = binom.logsf(k - 1, n, gamma)
    return np.clip(-logsf / np.log(10.0), 0.0, 300.0)


def green_window_features(
    df: pd.DataFrame,
    indicator_col: str,
    gamma: float,
    prefix: str,
    valid_col: str | None = None,
    window_sizes=WINDOW_SIZES,
    shapes=SHAPES,
) -> pd.DataFrame:
    """Per-token -log10(p) window features for one green-indicator column.

    valid_col: optional 0/1 column; invalid positions (e.g. KGW repeated
    ngrams) are excluded from both count and window length.
    Returns a DataFrame with new feature columns, aligned to df rows.
    Requires df sorted by (document_id, position).
    """
    out: dict[str, np.ndarray] = {}
    n_rows = len(df)
    for w in window_sizes:
        for shape in shapes:
            out[f"{prefix}_nlp_{shape}{w}"] = np.zeros(n_rows, dtype=np.float32)

    for _, sub in df.groupby("document_id", sort=False):
        sl = sub.index.to_numpy()
        x = sub[indicator_col].to_numpy().astype(np.int64)
        if valid_col is not None:
            v = sub[valid_col].to_numpy().astype(np.int64)
            xg, xn = x * v, v
        else:
            xg, xn = x, np.ones_like(x)
        for w in window_sizes:
            for shape in shapes:
                k, _ = _window_counts(xg, w, shape)
                n_eff, _ = _window_counts(xn, w, shape)
                nlp = np.zeros(len(x), dtype=np.float32)
                nz = n_eff > 0
                nlp[nz] = _neglog10_binom_sf(k[nz], n_eff[nz], gamma).astype(np.float32)
                out[f"{prefix}_nlp_{shape}{w}"][sl] = nlp

    return pd.DataFrame(out, index=df.index)


def _neglog10_gamma_sf(s: np.ndarray, n: np.ndarray, var_factor: float) -> np.ndarray:
    """-log10 P(S >= s) for a sum of n unit-mean scores with variance var_factor.

    Moment-matched Gamma as in textseal detector: shape n/var, scale var.
    For Gumbel (Exp(1) increments) var_factor=1 gives the exact Gamma(n,1) tail.
    """
    from scipy.special import gammaincc

    with np.errstate(divide="ignore"):
        p = gammaincc(n / var_factor, s / var_factor)
        return np.clip(-np.log10(np.maximum(p, 1e-300)), 0.0, 300.0)


def score_window_features(
    df: pd.DataFrame,
    score_col: str,
    prefix: str,
    var_factor: float = 1.0,
    valid_col: str | None = None,
    window_sizes=WINDOW_SIZES,
    shapes=SHAPES,
) -> pd.DataFrame:
    """Per-token -log10(p) window features for a continuous unit-mean score.

    NaN scores (unscored positions) are excluded; valid_col additionally
    restricts which positions count (e.g. v1 first-occurrence dedup).
    """
    out: dict[str, np.ndarray] = {}
    n_rows = len(df)
    for w in window_sizes:
        for shape in shapes:
            out[f"{prefix}_nlp_{shape}{w}"] = np.zeros(n_rows, dtype=np.float32)

    for _, sub in df.groupby("document_id", sort=False):
        sl = sub.index.to_numpy()
        s = sub[score_col].to_numpy().astype(np.float64)
        ok = ~np.isnan(s)
        if valid_col is not None:
            ok &= sub[valid_col].to_numpy().astype(bool)
        s_clean = np.where(ok, s, 0.0)
        cnt = ok.astype(np.int64)
        for w in window_sizes:
            for shape in shapes:
                ssum, _ = _window_counts(s_clean, w, shape)
                n_eff, _ = _window_counts(cnt, w, shape)
                nlp = np.zeros(len(s), dtype=np.float32)
                nz = n_eff > 0
                nlp[nz] = _neglog10_gamma_sf(ssum[nz], n_eff[nz], var_factor).astype(np.float32)
                out[f"{prefix}_nlp_{shape}{w}"][sl] = nlp

    return pd.DataFrame(out, index=df.index)


def max_evidence(features: pd.DataFrame) -> np.ndarray:
    """min-p over windows == max of -log10(p) columns."""
    return features.to_numpy().max(axis=1)
