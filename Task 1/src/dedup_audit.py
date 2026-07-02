"""Per-scheme dedup audit: v0 (none) / v1 (context) / v2 (context+token).

KEY CONTEXT: the parquet caches hold V1 flags (detectors/gumbel_textseal.py
computes context-only dedup; cache predates the v2 jump of sub_009). This
script recomputes all flag variants from token_ids and evaluates the full
per-scheme matrix. Expect all-v2 to reproduce sub_009/sub_010_lp val numbers.

Fit train / eval val, TPR@0.1%FPR. One-at-a-time from the all-v2 baseline,
then best combo; transitions (0.02,45) and (0.03,80) for the finalists.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hmm import fit_emissions, make_transitions, hmm_scores, SCHEMES  # noqa
from metrics import tpr_at_fpr  # noqa

C = Path(__file__).resolve().parent.parent / "cache"

GUM_N, TS_N, KGW_CTX = 2, 3, 3  # context sizes (kgw: 3 prior tokens, self-salt)


def flags_for_doc(ids: np.ndarray) -> dict[str, np.ndarray]:
    """All dedup-flag variants for one document."""
    n = len(ids)
    out = {}
    for name, ng, start in [("gum", GUM_N, GUM_N + 1), ("ts", TS_N, TS_N + 1), ("kgw", KGW_CTX, KGW_CTX)]:
        v0 = np.zeros(n, np.int8); v1 = np.zeros(n, np.int8); v2 = np.zeros(n, np.int8)
        seen1: set = set(); seen2: set = set()
        for p in range(start, n):
            ctx = tuple(ids[p - ng: p])
            v0[p] = 1
            if ctx not in seen1:
                seen1.add(ctx); v1[p] = 1
            ct = (ctx, ids[p])
            if ct not in seen2:
                seen2.add(ct); v2[p] = 1
        out[f"{name}_v0"], out[f"{name}_v1"], out[f"{name}_v2"] = v0, v1, v2
    # unigram: v0 = none, big = repeated-bigram (current), tok = token-dedup
    big = np.zeros(n, np.int8); tokd = np.zeros(n, np.int8)
    seenb: set = set(); seent: set = set()
    for p in range(n):
        if p >= 1:
            b = (ids[p - 1], ids[p])
            if b in seenb:
                big[p] = 1  # repeated -> invalid (matches is_repeated_ngram_2)
            seenb.add(b)
        if ids[p] in seent:
            tokd[p] = 1
        seent.add(ids[p])
    out["uni_rep_big"], out["uni_rep_tok"] = big, tokd
    return out


def build_flags(df: pd.DataFrame) -> dict[str, np.ndarray]:
    docs = df["document_id"].to_numpy()
    tid = df["token_id"].to_numpy().astype(np.int64)
    acc: dict[str, list] = {}
    start = 0
    for i in range(1, len(df) + 1):
        if i == len(df) or docs[i] != docs[start]:
            f = flags_for_doc(tid[start:i])
            for k, v in f.items():
                acc.setdefault(k, []).append(v)
            start = i
    return {k: np.concatenate(v) for k, v in acc.items()}


print("computing flag variants ...")
tr = pd.read_parquet(C / "train_signals.parquet")
tre = pd.read_parquet(C / "train_evidence.parquet")
va = pd.read_parquet(C / "validation_signals.parquet")
vae = pd.read_parquet(C / "validation_evidence.parquet")
ftr = build_flags(tr)
fva = build_flags(va)
y = va["label"].to_numpy()

# sanity: v1 flags should match the cache columns (cache is v1)
for cache_col, flag in [("gumbel_first", "gum_v1"), ("textseal_first", "ts_v1")]:
    same = (tr[cache_col].to_numpy() == ftr[flag]).mean()
    print(f"  sanity {cache_col} == {flag}: {same:.4f}")
# kgw_first from cluster used 4-gram incl token = v2 semantics
same = (tr["kgw_first"].to_numpy() == ftr["kgw_v2"]).mean()
print(f"  sanity kgw_first == kgw_v2: {same:.4f}")

BEST = {"unigram": "entropy", "kgw": "entropy", "gumbel": "logprob", "textseal": "logprob"}


def apply(df, flags, gum, ts, kgw, uni):
    d = df.copy()
    d["gumbel_first"] = flags[f"gum_{gum}"]
    d["textseal_first"] = flags[f"ts_{ts}"]
    d["kgw_first"] = flags[f"kgw_{kgw}"]
    if uni == "v0":
        d["is_repeated_ngram_2"] = 0
    elif uni == "big":
        d["is_repeated_ngram_2"] = flags["uni_rep_big"]
    elif uni == "tok":
        d["is_repeated_ngram_2"] = flags["uni_rep_tok"]
    return d


def evaluate(gum, ts, kgw, uni, pe=0.02, ms=45.0, tag=""):
    trd = apply(tr, ftr, gum, ts, kgw, uni)
    vad = apply(va, fva, gum, ts, kgw, uni)
    params = fit_emissions(trd, tre, use_entropy=BEST)
    logA, logpi = make_transitions(pe, ms)
    sc = hmm_scores(vad, params, logA, logpi)
    t01 = tpr_at_fpr(y, sc, 0.001)
    print(f"  {t01:.4f}   gum={gum} ts={ts} kgw={kgw} uni={uni} trans=({pe},{ms}) {tag}")
    return t01


print(f"\n{'T@0.1%':>8s}   combo")
print("— reference points —")
evaluate("v1", "v1", "v2", "big", tag="[= cache today / sub_006-era]")
base = evaluate("v2", "v2", "v2", "big", tag="[expect ~sub_009 0.4316]")
evaluate("v2", "v2", "v2", "big", 0.03, 80.0, tag="[expect ~sub_010_lp 0.4384]")

print("— one-at-a-time from all-v2 —")
results = {}
for g in ["v0", "v1", "v2"]:
    if g != "v2":
        results[("gum", g)] = evaluate(g, "v2", "v2", "big")
for t in ["v0", "v1"]:
    results[("ts", t)] = evaluate("v2", t, "v2", "big")
for k in ["v0", "v1"]:
    results[("kgw", k)] = evaluate("v2", "v2", k, "big")
for u in ["v0", "tok"]:
    results[("uni", u)] = evaluate("v2", "v2", "v2", u)
