"""v2 re-test sweep: fit on train, eval on val (TPR@0.1%FPR).

Re-tests ideas that were 'closed' under v1 dedup, now that the v2 cache
(ctx+token dedup) broke the plateau. Prints a ranked table.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hmm  # noqa: E402
from hmm import fit_emissions, make_transitions, hmm_scores  # noqa: E402
from metrics import tpr_at_fpr  # noqa: E402

CACHE = Path(__file__).resolve().parent.parent / "cache"

train_df = pd.read_parquet(CACHE / "train_signals.parquet")
train_ev = pd.read_parquet(CACHE / "train_evidence.parquet")
val_df = pd.read_parquet(CACHE / "validation_signals.parquet")
y = val_df["label"].to_numpy()

BEST = {"unigram": "entropy", "kgw": "entropy", "gumbel": "logprob", "textseal": "logprob"}


def evaluate(cond, p_enter, mean_span, eps=0.5, label=""):
    params = fit_emissions(train_df, train_ev, eps=eps, use_entropy=cond)
    logA, logpi = make_transitions(p_enter, mean_span)
    sc = hmm_scores(val_df, params, logA, logpi)
    t01 = tpr_at_fpr(y, sc, 0.001)
    t1 = tpr_at_fpr(y, sc, 0.01)
    return t01, t1


def run(label, cond, p_enter=0.03, mean_span=80.0, eps=0.5):
    t01, t1 = evaluate(cond, p_enter, mean_span, eps, label)
    results.append((t01, t1, label))
    print(f"  {t01:.4f}  {t1:.4f}   {label}")


results: list = []

print("== BASELINE REPRODUCTION (expect sub_009~0.4316 @ default, lp~0.4384 @ 0.03/80) ==")
print("  T@0.1% T@1%    config")
run("base best, trans (0.02,45)  [~sub_009]", BEST, 0.02, 45.0)
run("base best, trans (0.03,80)  [~sub_010_lp]", BEST, 0.03, 80.0)
run("rank gum+ts, trans (0.03,80) [~sub_010_rank]",
    {**BEST, "gumbel": "rank", "textseal": "rank"}, 0.03, 80.0)

print("\n== CONDITIONING VARIANTS (trans 0.03/80) ==")
run("gum rank + ts logprob", {**BEST, "gumbel": "rank"})
run("gum logprob + ts rank", {**BEST, "textseal": "rank"})
run("gum both + ts both", {**BEST, "gumbel": "both", "textseal": "both"})
run("kgw logprob (rest best)", {**BEST, "kgw": "logprob"})
run("unigram pgreen (rest best)", {**BEST, "unigram": "pgreen"})
run("ts dual_lp (rest best)", {**BEST, "textseal": "dual_lp"})
run("gum lp_match + ts lp_match", {**BEST, "gumbel": "lp_match", "textseal": "lp_match"})
run("all entropy (legacy sub_005)", {s: "entropy" for s in hmm.SCHEMES})

print("\n== EPS SWEEP (rank config, trans 0.03/80) ==")
RANK = {**BEST, "gumbel": "rank", "textseal": "rank"}
for eps in (0.1, 0.25, 0.5, 1.0, 2.0):
    run(f"rank, eps={eps}", RANK, eps=eps)

print("\n== TRANSITION GRID (rank config, eps 0.5) ==")
for pe in (0.02, 0.025, 0.03, 0.035, 0.04):
    for ms in (50.0, 65.0, 80.0, 95.0, 110.0):
        run(f"rank, trans ({pe},{ms})", RANK, pe, ms)

print("\n== TOP 15 OVERALL ==")
for t01, t1, label in sorted(results, reverse=True)[:15]:
    print(f"  {t01:.4f}  {t1:.4f}   {label}")
