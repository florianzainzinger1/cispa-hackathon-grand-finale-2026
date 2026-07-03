"""Ranking-based per-scheme sanity gate (strategy §1).

For one scheme's green indicator:
  1. window features -> max evidence per token
  2. pooled tpr_at_fpr on the labeled split (per scheme; ceiling ~= the
     scheme's share of spans, since schemes don't detect each other's spans)
  3. top-k evidence enrichment in label=1
  4. green-rate check inside top-evidence windows vs gamma

Usage: python src/sanity_check.py --scheme unigram --gamma 0.5 --split train
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import green_window_features, max_evidence  # noqa: E402
from metrics import tpr_at_fpr  # noqa: E402

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"

SCHEME_COLS = {
    "unigram": ("unigram_green", None),
    "kgw": ("kgw_green", "kgw_valid"),
}


def run(scheme: str, gamma: float, split: str) -> None:
    df = pd.read_parquet(CACHE_DIR / f"{split}_signals.parquet")
    indicator_col, valid_col = SCHEME_COLS[scheme]
    if indicator_col not in df.columns:
        sys.exit(f"column {indicator_col} not in cache — run the {scheme} extractor first")

    y = df["label"].to_numpy()
    green = df[indicator_col].to_numpy()

    print(f"=== sanity gate: {scheme} on {split} (gamma={gamma}) ===")
    print(f"tokens: {len(df):,}, label=1: {y.mean():.3f}")
    print(f"green rate overall:   {green.mean():.4f} (null expectation {gamma})")
    print(f"green rate | label=0: {green[y == 0].mean():.4f}")
    print(f"green rate | label=1: {green[y == 1].mean():.4f}  "
          "(mixes ALL schemes; mild elevation is expected, absence is a bug)")

    feats = green_window_features(df, indicator_col, gamma, scheme, valid_col)
    ev = max_evidence(feats)

    score = tpr_at_fpr(y, ev, 0.001)
    print(f"\nper-scheme pooled TPR@0.1%FPR: {score:.4f}")
    for w in (16, 32, 64, 128):
        cols = [c for c in feats.columns if c.endswith(f"center{w}")]
        s = tpr_at_fpr(y, feats[cols[0]].to_numpy(), 0.001)
        print(f"  centered window {w:4d} alone: {s:.4f}")

    for k in (1000, 5000, 20000):
        top = np.argsort(-ev)[:k]
        print(f"top-{k} evidence tokens: label=1 rate = {y[top].mean():.3f}, "
              f"green rate = {green[top].mean():.3f}")

    # what sits in the clean tail?
    clean = df[y == 0].copy()
    clean["evidence"] = ev[y == 0]
    tail = clean.nlargest(15, "evidence")
    print("\ntop-15 clean (label=0) tokens by evidence — the FP tail:")
    print(tail[["document_id", "position", "evidence", "is_special_token",
                "is_repeated_ngram_3", "is_number_like", "is_newline_like"]]
          .to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scheme", required=True, choices=list(SCHEME_COLS))
    p.add_argument("--gamma", type=float, required=True)
    p.add_argument("--split", default="train", choices=["train", "validation"])
    a = p.parse_args()
    run(a.scheme, a.gamma, a.split)
