"""Load the watermark-localization JSONL splits.

token_ids is the authoritative token sequence — never re-tokenize `text`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SPLIT_FILES = {
    "train": DATA_DIR / "train.jsonl",
    "validation": DATA_DIR / "validation.jsonl",
    "test": DATA_DIR / "test.jsonl",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"File does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def load_split(split: str, data_dir: Path | None = None) -> list[dict[str, Any]]:
    """Load one of train / validation / test."""
    if split not in SPLIT_FILES:
        raise ValueError(f"Unknown split {split!r}, expected one of {list(SPLIT_FILES)}")
    path = SPLIT_FILES[split] if data_dir is None else Path(data_dir) / f"{split}.jsonl"
    rows = read_jsonl(path)
    for row in rows:
        n = len(row["token_ids"])
        if len(row["token_pieces"]) != n:
            raise ValueError(f"{split} {row['document_id']}: token_pieces length mismatch")
        if "labels" in row and len(row["labels"]) != n:
            raise ValueError(f"{split} {row['document_id']}: labels length mismatch")
    return rows


def spans_from_labels(labels: list[int]) -> list[tuple[int, int]]:
    """Contiguous label-1 spans as (start, end) with end exclusive."""
    spans: list[tuple[int, int]] = []
    start = None
    for i, v in enumerate(labels):
        if v == 1 and start is None:
            start = i
        elif v == 0 and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(labels)))
    return spans
