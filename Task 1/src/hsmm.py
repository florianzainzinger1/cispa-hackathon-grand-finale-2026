"""Explicit-duration HSMM decode: clean self-loop + watermark segments with an
explicit span-duration prior. Emissions come from the champion (src/hmm.py:
fit_emissions / emission_loglik); this module ONLY replaces the decode.

Generative model per document (T tokens):
  t=0: clean w.p. 1-p0, else (p0/4 per scheme) inside a span whose remaining
       length r >= 1 follows the residual prior res(r) proportional to P(D >= r).
  clean t -> t+1: stay clean w.p. 1-pe, else (pe/4 per scheme) a segment with
       duration d ~ P(D) covers t+1..t+d; the token after a segment is clean
       w.p. 1 (true spans are separated by >= 12 clean tokens in the data).
  Right-censoring: a segment whose observed part is cut off by the doc end
       costs P(D >= d_observed) (survival), not P(D = d_observed).

Score per token = -log10 P(clean_t | doc signals), via segment-level
forward-backward. With a geometric duration prior this reduces EXACTLY to the
champion 5-state HMM with make_transitions(pe, mean) — asserted in
src/eval_hsmm_gate.py.

z_mode:
  "fixed" (default): Z = paths ending clean at T-1  +  paths ending inside a
      right-censored segment (+ whole-doc-in-span when T <= dmax; never
      triggers on this data, all docs >= 513 tokens > dmax).
  "ref": bug-compatible with the fresh-context researcher's reference
      (scratchpad hsmm_gate.py): its Z additionally re-counts every
      doc-starts-inside-a-span path (those are already inside alpha via the
      p0 branch of the alpha recursion), inflating Z by up to 2x in docs whose
      start is confidently watermarked; its residual prior is also normalized
      including a spurious r=0 term. Kept only to reproduce reference numbers.
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hmm import emission_loglik  # noqa: E402

NEG = -1e30

# duration-prior defaults: train spans live in [24, 320] with a pile-up at the
# generation cap 320 (38/396 train spans); kappa is the add-count smoothing
# INSIDE the support. Support outside [lo, hi] gets zero mass by design — the
# cap is a property of the generator (train AND validation both max out at
# exactly 320), not a sampling artifact.
D_LO, D_HI, D_MAX = 24, 320, 340


def span_lengths(fit_df: pd.DataFrame) -> np.ndarray:
    """Lengths of contiguous label-1 runs (within a document) in fit_df."""
    y = fit_df["label"].to_numpy()
    dd = fit_df["document_id"].to_numpy()
    out = []
    i, n = 0, len(y)
    while i < n:
        if y[i] == 1:
            j = i
            while j < n and y[j] == 1 and dd[j] == dd[i]:
                j += 1
            out.append(j - i)
            i = j
        else:
            i += 1
    return np.asarray(out, dtype=np.int64)


def duration_logpmf_from(
    fit_df: pd.DataFrame,
    dmax: int = D_MAX,
    lo: int = D_LO,
    hi: int = D_HI,
    kappa: float = 0.25,
    geom_blend: float = 0.0,
    geom_mean: float = 80.0,
) -> np.ndarray:
    """Smoothed empirical span-duration log-pmf from the FIT split's labels.

    Support [lo, hi] (widened if the fit split contains spans outside it, so
    no observed span ever gets zero prior mass); add-kappa smoothing inside
    the support; optional blend with a truncated geometric on the support.
    """
    sl = span_lengths(fit_df)
    if len(sl) == 0:
        raise ValueError("fit split contains no labeled spans")
    lo = min(lo, int(sl.min()))
    hi = max(hi, int(sl.max()))
    if hi > dmax:
        dmax = hi + 20
    cnt = np.bincount(sl, minlength=dmax + 1).astype(np.float64)
    pmf = np.zeros(dmax + 1)
    pmf[lo : hi + 1] = cnt[lo : hi + 1] + kappa
    pmf /= pmf.sum()
    if geom_blend > 0.0:
        g = np.zeros(dmax + 1)
        d = np.arange(lo, hi + 1, dtype=np.float64)
        g[lo : hi + 1] = (1.0 - 1.0 / geom_mean) ** (d - lo) / geom_mean
        g[lo : hi + 1] /= g[lo : hi + 1].sum()
        pmf = (1.0 - geom_blend) * pmf + geom_blend * g
    return np.log(np.maximum(pmf, 1e-300))


def geometric_logpmf(dmax: int = 2000, mean: float = 80.0) -> np.ndarray:
    """Geometric duration log-pmf, support d >= 1 (matches the champion HMM
    with make_transitions(pe, mean): P(D=d) = (1/mean)(1-1/mean)^(d-1)."""
    d = np.arange(dmax + 1, dtype=np.float64)
    pmf = np.zeros(dmax + 1)
    pmf[1:] = (1.0 / mean) * (1.0 - 1.0 / mean) ** (d[1:] - 1.0)
    return np.log(np.maximum(pmf, 1e-300))


def hsmm_doc(
    ll: np.ndarray,
    logPD: np.ndarray,
    pe: float = 0.03,
    p0: float = 0.07,
    z_mode: str = "fixed",
) -> np.ndarray:
    """P(clean_t | doc) for one document. ll: (T, 5) emission log-liks
    (col 0 = clean, cols 1..4 = schemes). logPD: (dmax+1,) duration log-pmf."""
    T = ll.shape[0]
    dmax = len(logPD) - 1
    ll0 = np.ascontiguousarray(ll[:, 0])
    cs = np.zeros((T + 1, 4))
    np.cumsum(ll[:, 1:5], axis=0, out=cs[1:])

    pd_ = np.exp(logPD)
    surv_p = np.maximum(1.0 - np.cumsum(pd_) + pd_, 0.0)  # P(D >= d)
    lsurv = np.log(np.maximum(surv_p, 1e-300))
    if z_mode == "ref":
        # reference normalizes the residual over r = 0..dmax (incl. bogus r=0)
        res = surv_p / surv_p.sum()
    else:
        res = surv_p.copy()
        res[0] = 0.0
        res /= res.sum()
    lres = np.log(np.maximum(res, 1e-300))

    lpe = np.log(pe / 4.0)
    lstay = np.log1p(-pe)
    lp0 = np.log(p0 / 4.0)
    lpc0 = np.log1p(-p0)

    ds = np.arange(dmax + 1)

    # ---- forward: alpha[t] = log P(obs[0..t], token t clean) ----
    alpha = np.full(T, NEG)
    alpha[0] = lpc0 + ll0[0]
    for t in range(1, T):
        stay = alpha[t - 1] + lstay
        m = stay
        acc_parts = []
        dm = min(dmax, t - 1)
        if dm >= 1:
            d = ds[1 : dm + 1]
            seg = cs[t] - cs[t - d]  # (dm, 4): emissions of tokens t-d..t-1
            vals = (alpha[t - 1 - d] + lpe + logPD[d])[:, None] + seg
            vm = vals.max()
            if vm > m:
                m = vm
            acc_parts.append(vals)
        if t <= dmax:
            si = lp0 + lres[t] + cs[t]  # span covers 0..t-1 (cs[0] = 0)
            vm = si.max()
            if vm > m:
                m = vm
            acc_parts.append(si)
        s = np.exp(stay - m)
        for part in acc_parts:
            s += np.exp(part - m).sum()
        alpha[t] = ll0[t] + m + np.log(s)

    # ---- backward: beta[t] = log P(obs[t+1..T-1] | token t clean) ----
    beta = np.full(T, NEG)
    beta[T - 1] = 0.0
    for t in range(T - 2, -1, -1):
        stay = lstay + ll0[t + 1] + beta[t + 1]
        m = stay
        s = None
        dm = min(dmax, T - 1 - t)
        d = ds[1 : dm + 1]
        ends = t + d  # segment covers t+1..t+d
        seg = cs[ends + 1] - cs[t + 1]  # (dm, 4)
        k = dm - 1 if ends[-1] == T - 1 else dm  # censored tail?
        vals = np.full((dm, 4), NEG)
        if k > 0:
            vals[:k] = (lpe + logPD[d[:k]] + ll0[ends[:k] + 1] + beta[ends[:k] + 1])[
                :, None
            ] + seg[:k]
        if k < dm:
            vals[k] = lpe + lsurv[d[k]] + seg[k]  # segment reaches doc end
        vm = vals.max()
        if vm > m:
            m = vm
        beta[t] = m + np.log(np.exp(stay - m) + np.exp(vals - m).sum())

    # ---- partition function ----
    z_parts = [np.array([alpha[T - 1]])]
    dm = min(dmax, T - 1)
    d = ds[1 : dm + 1]
    seg = cs[T] - cs[T - d]  # (dm, 4)
    z_parts.append(((alpha[T - 1 - d] + lpe + lsurv[d])[:, None] + seg).ravel())
    if z_mode == "ref":
        # reference bug: re-counts start-inside paths (already inside alpha)
        dm2 = min(dmax, T)
        d2 = ds[1 : dm2 + 1]
        seg2 = cs[np.minimum(d2, T)]  # cs[0] = 0
        v = np.full((dm2, 4), NEG)
        inner = d2 <= T - 1
        if inner.any():
            di = d2[inner]
            v[inner] = (lp0 + lres[di] + ll0[di] + beta[di])[:, None] + seg2[inner]
        whole = d2 == T
        if whole.any():
            v[whole] = lp0 + lres[min(T, dmax)] + seg2[whole]
        z_parts.append(v.ravel())
    elif T <= dmax:
        # whole doc inside one span, censored at both ends
        tail = res[T:].sum()
        z_parts.append(lp0 + np.log(max(tail, 1e-300)) + cs[T])
    zcat = np.concatenate(z_parts)
    m = zcat.max()
    logZ = m + np.log(np.exp(zcat - m).sum())

    return np.clip(np.exp(alpha + beta - logZ), 1e-30, 1.0)


def _doc_neglog10(args) -> np.ndarray:
    ll, logPD, pe, p0, z_mode = args
    return -np.log10(hsmm_doc(ll, logPD, pe, p0, z_mode))


def hsmm_scores(
    df: pd.DataFrame,
    params: dict,
    logPD: np.ndarray,
    pe: float = 0.03,
    p0: float = 0.07,
    z_mode: str = "fixed",
    processes: int | None = None,
) -> np.ndarray:
    """-log10 P(clean | signals) per token, all documents in df.

    params: champion emission params (hmm.fit_emissions on the FIT split).
    logPD: duration log-pmf (duration_logpmf_from on the SAME fit split).
    """
    ll = emission_loglik(df, params)
    docs = df["document_id"].to_numpy()
    bounds = np.r_[0, np.where(docs[1:] != docs[:-1])[0] + 1, len(docs)]
    jobs = [
        (ll[a:b], logPD, pe, p0, z_mode) for a, b in zip(bounds[:-1], bounds[1:])
    ]
    if processes is not None and processes > 1:
        with Pool(processes) as pool:
            outs = pool.map(_doc_neglog10, jobs, chunksize=8)
    else:
        outs = [_doc_neglog10(j) for j in jobs]
    return np.concatenate(outs)
