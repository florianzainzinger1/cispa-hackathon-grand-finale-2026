"""Label EDA on train/validation: span lengths, spans/doc, gaps, positions.

Writes reports/03_label_eda.md. These statistics feed span filters (minimum
run length) and HMM transition priors later.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_data import load_split, spans_from_labels  # noqa: E402

REPORT_PATH = Path(__file__).resolve().parent.parent / "reports" / "03_label_eda.md"


def pct(values, q):
    return float(np.percentile(np.asarray(values), q))


def analyze_split(split: str) -> dict:
    rows = load_split(split)
    doc_lens, span_lens, gaps, spans_per_doc = [], [], [], []
    rel_starts, rel_ends = [], []
    starts_at_0 = ends_at_len = 0
    label_counts = Counter()

    for row in rows:
        labels = row["labels"]
        n = len(labels)
        doc_lens.append(n)
        label_counts.update(labels)
        spans = spans_from_labels(labels)
        spans_per_doc.append(len(spans))
        for k, (a, b) in enumerate(spans):
            span_lens.append(b - a)
            rel_starts.append(a / n)
            rel_ends.append(b / n)
            if a == 0:
                starts_at_0 += 1
            if b == n:
                ends_at_len += 1
            if k + 1 < len(spans):
                gaps.append(spans[k + 1][0] - b)

    return {
        "rows": rows,
        "n_docs": len(rows),
        "doc_lens": doc_lens,
        "span_lens": span_lens,
        "gaps": gaps,
        "spans_per_doc": spans_per_doc,
        "rel_starts": rel_starts,
        "rel_ends": rel_ends,
        "starts_at_0": starts_at_0,
        "ends_at_len": ends_at_len,
        "n_pos": label_counts[1],
        "n_neg": label_counts[0],
    }


def dist_line(name, values) -> str:
    if not values:
        return f"| {name} | – | – | – | – | – | – |"
    v = np.asarray(values)
    return (
        f"| {name} | {len(v)} | {v.min()} | {pct(v,25):.0f} | "
        f"{np.median(v):.0f} | {pct(v,75):.0f} | {v.max()} |"
    )


def main() -> None:
    out = ["# Label EDA — train & validation", ""]
    for split in ("train", "validation"):
        a = analyze_split(split)
        n_tok = a["n_pos"] + a["n_neg"]
        out += [
            f"## {split}",
            "",
            f"- Documents: **{a['n_docs']}**, tokens: **{n_tok:,}**, "
            f"watermarked: **{a['n_pos']:,} ({100*a['n_pos']/n_tok:.1f}%)**",
            f"- Docs without any watermark span: "
            f"**{sum(1 for s in a['spans_per_doc'] if s == 0)}**",
            f"- Spans starting at doc position 0: **{a['starts_at_0']}**, "
            f"spans ending at doc end: **{a['ends_at_len']}** "
            f"(of {len(a['span_lens'])} spans)",
            "",
            "| distribution | n | min | p25 | median | p75 | max |",
            "|---|---|---|---|---|---|---|",
            dist_line("doc length (tokens)", a["doc_lens"]),
            dist_line("span length (tokens)", a["span_lens"]),
            dist_line("gap between spans", a["gaps"]),
            dist_line("spans per doc", a["spans_per_doc"]),
            "",
            "Spans per doc histogram: "
            + ", ".join(
                f"{k} spans: {v}"
                for k, v in sorted(Counter(a["spans_per_doc"]).items())
            ),
            "",
            "Relative span start positions (fraction of doc): "
            + ", ".join(f"{q}%={pct(a['rel_starts'], q):.2f}" for q in (5, 25, 50, 75, 95)),
            "",
        ]

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    print(f"\nwritten to {REPORT_PATH}")


if __name__ == "__main__":
    main()
