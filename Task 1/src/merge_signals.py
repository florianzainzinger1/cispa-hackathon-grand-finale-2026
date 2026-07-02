"""Merge extractor outputs (npz from cluster) into the Parquet signal cache.

npz layout: lengths (per doc, jsonl file order) + one or more flat int8/float32
arrays. Alignment is by document order + token position, verified against the
cache's per-document lengths.

Usage: python src/merge_signals.py --split train --npz cache/kgw_train.npz --cols kgw_green kgw_first
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_data import load_split  # noqa: E402

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"


def merge(split: str, npz_path: Path, cols: list[str]) -> None:
    cache_path = CACHE_DIR / f"{split}_signals.parquet"
    df = pd.read_parquet(cache_path)
    data = np.load(npz_path)

    rows = load_split(split)
    lengths = data["lengths"]
    if len(lengths) != len(rows):
        raise ValueError(f"doc count mismatch: npz {len(lengths)} vs jsonl {len(rows)}")
    for row, n in zip(rows, lengths):
        if len(row["token_ids"]) != n:
            raise ValueError(f"{row['document_id']}: length mismatch npz={n}")

    n_total = int(lengths.sum())
    if len(df) != n_total:
        raise ValueError(f"cache rows {len(df)} != npz tokens {n_total}")

    # cache row order == jsonl doc order + position (build_cache guarantees it)
    expected_docs = np.repeat([r["document_id"] for r in rows], lengths)
    if not (df["document_id"].to_numpy() == expected_docs).all():
        raise ValueError("document order mismatch between cache and jsonl")

    for col in cols:
        df[col] = data[col]
    df.to_parquet(cache_path, index=False)
    print(f"{split}: merged {cols} into {cache_path} ({len(df):,} rows)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True)
    p.add_argument("--npz", type=Path, required=True)
    p.add_argument("--cols", nargs="+", required=True)
    a = p.parse_args()
    merge(a.split, a.npz, a.cols)
