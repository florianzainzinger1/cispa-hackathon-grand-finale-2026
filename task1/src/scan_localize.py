"""Scan-statistic localization — port of TextSeal repo `localized_detect` semantics.

Tests hypotheses #2+#4 (native localized detection / interval scan statistic):
dyadic windows (2^p, stride len/2, repo `_geometric_cover_search`), token score
= max window z over all windows containing the token. Evidence = HMM emission
LLR (all 4 schemes + champion conditioning), so the scan gets our best
per-token signal. Null (mu0, var0) estimated on TRAIN clean tokens.

Variants: scan alone, scan+champion fusion (rank-mean, max of train-calibrated
quantiles), kgw-channel-only scan (does span-sum aggregation lift weak KGW
spans past the global threshold?).

Clean protocol: fit train, eval validation, TPR@0.1%FPR.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hmm  # noqa
from hmm import (SCHEMES, emission_loglik, fit_emissions, hmm_scores,
                 make_transitions)  # noqa
from metrics import tpr_at_fpr  # noqa

C = Path(__file__).resolve().parent.parent / "cache"

BEST = {"unigram": "entropy", "kgw": "entropy", "gumbel": "logprob", "textseal": "logprob"}

tr = pd.read_parquet(C / "train_signals.parquet")
tre = pd.read_parquet(C / "train_evidence.parquet")
va = pd.read_parquet(C / "validation_signals.parquet")
vae = pd.read_parquet(C / "validation_evidence.parquet")
y = va["label"].to_numpy()
wm = y == 1
dom = vae[SCHEMES].to_numpy().argmax(axis=1)
kgw_dom = wm & (dom == 1)

params = fit_emissions(tr, tre, use_entropy=BEST)
logA, logpi = make_transitions(0.03, 80.0)

champ = hmm_scores(va, params, logA, logpi)
t_champ = tpr_at_fpr(y, champ, 0.001)
tau = np.quantile(champ[y == 0], 1 - 0.001)
print(f"champion HMM posterior:      {t_champ:.4f}  kgw-recall "
      f"{(kgw_dom & (champ >= tau)).sum() / kgw_dom.sum():.3f}")


def token_llr(df: pd.DataFrame, schemes=None) -> np.ndarray:
    """Per-token LLR log P(obs|wm-mix) - log P(obs|clean); missing obs -> 0."""
    ll = emission_loglik(df, params)
    if schemes is not None:
        # rebuild with only the chosen schemes' channels
        obs = hmm.extract_observations(df, params.get("use_entropy", False))
        ll = np.zeros((len(df), 5), dtype=np.float64)
        for si, scheme in enumerate(SCHEMES):
            if scheme not in schemes:
                continue
            codes = obs[scheme]
            ok = codes >= 0
            lc = params[scheme]["log_clean"][codes[ok]]
            lw = params[scheme]["log_wm"][codes[ok]]
            for state in range(5):
                ll[ok, state] += np.where(state == si + 1, lw, lc)
    mix = np.logaddexp.reduce(ll[:, 1:] + np.log(0.25), axis=1)
    return mix - ll[:, 0]


def scan_scores(df: pd.DataFrame, llr: np.ndarray, mu0: float, var0: float,
                min_len: int = 16) -> np.ndarray:
    """Token score = max dyadic-window z over windows containing the token."""
    docs = df["document_id"].to_numpy()
    out = np.full(len(df), -np.inf)
    start = 0
    for i in range(1, len(df) + 1):
        if i == len(df) or docs[i] != docs[start]:
            x = llr[start:i]
            n = len(x)
            best = np.full(n, -np.inf)
            prefix = np.concatenate([[0.0], np.cumsum(x)])
            p = 0
            while (1 << p) <= n:
                length = 1 << p
                p += 1
                if length < min_len:
                    continue
                stride = max(1, length // 2)
                for s in range(0, n - length + 1, stride):
                    z = (prefix[s + length] - prefix[s] - length * mu0) / np.sqrt(length * var0)
                    seg = best[s:s + length]
                    np.maximum(seg, z, out=seg)
            # tokens never covered (doc shorter than min_len): whole-doc window
            if n < min_len and n > 0:
                z = (prefix[n] - n * mu0) / np.sqrt(n * var0)
                np.maximum(best, z, out=best)
            out[start:i] = best
            start = i
    return out


def report(tag: str, sc: np.ndarray) -> None:
    t01 = tpr_at_fpr(y, sc, 0.001)
    t = np.quantile(sc[y == 0], 1 - 0.001)
    kr = (kgw_dom & (sc >= t)).sum() / kgw_dom.sum()
    print(f"{tag:28s} {t01:.4f}  kgw-recall {kr:.3f}")


# null moments from TRAIN clean tokens
llr_tr = token_llr(tr)
clean_tr = tr["label"].to_numpy() == 0
mu0, var0 = llr_tr[clean_tr].mean(), llr_tr[clean_tr].var()
print(f"train clean llr: mu0={mu0:.4f} var0={var0:.4f}")

llr_va = token_llr(va)
for ml in (16, 24, 32, 64):
    report(f"scan all-schemes min_len={ml}", scan_scores(va, llr_va, mu0, var0, ml))

# fusion with champion: train-clean-calibrated quantile of each stat, then max/mean
sc_scan = scan_scores(va, llr_va, mu0, var0, 24)
llr_tr_scan = scan_scores(tr, llr_tr, mu0, var0, 24)
champ_tr = hmm_scores(tr, params, logA, logpi)


def train_quantile(stat_tr: np.ndarray, clean: np.ndarray, stat_va: np.ndarray) -> np.ndarray:
    ref = np.sort(stat_tr[clean])
    return np.searchsorted(ref, stat_va) / len(ref)


q_champ = train_quantile(champ_tr, clean_tr, champ)
q_scan = train_quantile(llr_tr_scan, clean_tr, sc_scan)
report("fusion max(qC,qS)", np.maximum(q_champ, q_scan))
report("fusion mean(qC,qS)", 0.5 * (q_champ + q_scan))
report("fusion 0.8*qC+0.2*qS", 0.8 * q_champ + 0.2 * q_scan)

# kgw-only channel scan: can span-sums lift weak KGW spans?
llr_kgw_tr = token_llr(tr, schemes={"kgw"})
mu0k, var0k = llr_kgw_tr[clean_tr].mean(), llr_kgw_tr[clean_tr].var()
llr_kgw_va = token_llr(va, schemes={"kgw"})
for ml in (24, 64):
    report(f"scan kgw-only min_len={ml}", scan_scores(va, llr_kgw_va, mu0k, var0k, ml))
sc_kscan = scan_scores(va, llr_kgw_va, mu0k, var0k, 64)
kscan_tr = scan_scores(tr, llr_kgw_tr, mu0k, var0k, 64)
q_kscan = train_quantile(kscan_tr, clean_tr, sc_kscan)
report("fusion max(qC,qKGWscan)", np.maximum(q_champ, q_kscan))
