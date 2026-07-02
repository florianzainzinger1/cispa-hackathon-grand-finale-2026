"""Build the per-token signal cache: cache/{split}_signals.parquet.

One row per token. Base + degenerate-token columns are written from day one
(they explain clean-tail false positives later). Scheme signal columns are
added by the individual detector runs (unigram here on CPU; kgw/gumbel/
textseal jobs merge into the same files later).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_data import load_split  # noqa: E402
from detectors.unigram import VOCAB_SIZE_CANDIDATES, green_indicator, green_mask  # noqa: E402

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
FIRST_SPECIAL_ID = 151643  # everything >= this is added/special/reserved in Qwen2.5

_NUMBER_RE = re.compile(r"^[\sĠ]*[0-9][0-9.,:%/\-]*$")


def repeated_ngram_flags(token_ids: list[int], n: int) -> np.ndarray:
    """1 if the n-gram ENDING at position i occurred earlier in the document."""
    flags = np.zeros(len(token_ids), dtype=np.int8)
    seen: set[tuple[int, ...]] = set()
    for i in range(len(token_ids)):
        if i + 1 < n:
            continue
        gram = tuple(token_ids[i + 1 - n : i + 1])
        if gram in seen:
            flags[i] = 1
        seen.add(gram)
    return flags


def base_columns(row: dict) -> pd.DataFrame:
    ids = row["token_ids"]
    pieces = row["token_pieces"]
    n = len(ids)
    ids_arr = np.asarray(ids, dtype=np.int64)
    df = pd.DataFrame(
        {
            "document_id": np.repeat(row["document_id"], n),
            "position": np.arange(n, dtype=np.int32),
            "token_id": ids_arr,
            "is_special_token": (ids_arr >= FIRST_SPECIAL_ID).astype(np.int8),
            "token_piece_length": np.asarray([len(p) for p in pieces], dtype=np.int16),
            "is_number_like": np.asarray(
                [1 if _NUMBER_RE.match(p) else 0 for p in pieces], dtype=np.int8
            ),
            "is_newline_like": np.asarray(
                [1 if ("Ċ" in p or p.strip("Ġ ") == "") else 0 for p in pieces],
                dtype=np.int8,
            ),
            "is_repeated_ngram_2": repeated_ngram_flags(ids, 2),
            "is_repeated_ngram_3": repeated_ngram_flags(ids, 3),
            "is_repeated_ngram_4": repeated_ngram_flags(ids, 4),
        }
    )
    if "labels" in row:
        df["label"] = np.asarray(row["labels"], dtype=np.int8)
    return df


def build_split(split: str) -> Path:
    rows = load_split(split)
    vocab_size = VOCAB_SIZE_CANDIDATES[0]
    mask = green_mask(vocab_size)

    frames = []
    for row in rows:
        df = base_columns(row)
        df["unigram_green"] = green_indicator(row["token_ids"], mask)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{split}_signals.parquet"
    out.to_parquet(path, index=False)
    print(f"{split}: {len(out):,} tokens -> {path}")
    return path


if __name__ == "__main__":
    splits = sys.argv[1:] or ["train", "validation", "test"]
    for split in splits:
        build_split(split)
