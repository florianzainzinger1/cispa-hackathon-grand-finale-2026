"""Standalone re-validation: load a saved submission.pt and re-run every check
the organizer's evaluator applies. Exit non-zero on any failure."""
import sys
import torch

import utils as U
from build_submission import validate_submission


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else U.work_path("outputs", "submission.pt")
    sub = torch.load(path, weights_only=False)
    print("loaded", path)
    print("keys:", sorted(sub.keys()))
    errs = validate_submission(sub)
    for m in U.MODELS:
        t = sub[m]
        # simple duplicate check within a model (exact dup rows)
        flat = t.reshape(t.shape[0], -1)
        uniq = torch.unique(flat, dim=0).shape[0]
        print(f"  {m:8} shape={tuple(t.shape)} dtype={t.dtype} "
              f"range=[{t.min():.3f},{t.max():.3f}] unique={uniq}/128")
    if errs:
        print("\nINVALID:")
        for e in errs:
            print("  -", e)
        sys.exit(1)
    print("\nVALID: submission passes all checks.")


if __name__ == "__main__":
    main()
