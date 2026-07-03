"""Local submission validator — mirrors the server-side rules.

Usage: python src/validate_submission.py --submission submissions/foo.jsonl [--test data/test.jsonl]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_data import read_jsonl  # noqa: E402


def validate(submission_rows, test_rows) -> None:
    expected_lengths = {r["document_id"]: len(r["token_ids"]) for r in test_rows}
    seen: set[str] = set()

    if len(submission_rows) != len(test_rows):
        raise ValueError(
            f"expected {len(test_rows)} rows, got {len(submission_rows)}"
        )

    for idx, row in enumerate(submission_rows, start=1):
        if set(row.keys()) != {"document_id", "scores"}:
            raise ValueError(f"row {idx}: must contain exactly document_id and scores")
        doc_id = row["document_id"]
        scores = row["scores"]
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError(f"row {idx}: invalid document_id")
        if doc_id in seen:
            raise ValueError(f"duplicate document_id {doc_id!r}")
        seen.add(doc_id)
        if doc_id not in expected_lengths:
            raise ValueError(f"unexpected document_id {doc_id!r}")
        if not isinstance(scores, list):
            raise ValueError(f"{doc_id}: scores must be a list")
        if len(scores) != expected_lengths[doc_id]:
            raise ValueError(
                f"{doc_id}: expected {expected_lengths[doc_id]} scores, got {len(scores)}"
            )
        for j, sc in enumerate(scores):
            if isinstance(sc, bool) or not isinstance(sc, (int, float)):
                raise ValueError(f"{doc_id}: scores[{j}] not numeric")
            f = float(sc)
            if not math.isfinite(f) or not 0.0 <= f <= 1.0:
                raise ValueError(f"{doc_id}: scores[{j}] out of [0,1]: {f}")

    missing = set(expected_lengths) - seen
    if missing:
        raise ValueError(f"missing {len(missing)} documents, e.g. {sorted(missing)[:5]}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument(
        "--test",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "test.jsonl",
    )
    args = parser.parse_args()

    submission_rows = read_jsonl(args.submission)
    test_rows = read_jsonl(args.test)
    validate(submission_rows, test_rows)
    n_scores = sum(len(r["scores"]) for r in submission_rows)
    print(f"OK: {len(submission_rows)} documents, {n_scores} scores — submission is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
