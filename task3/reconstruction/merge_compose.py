"""Compose an ISOLATED candidate = current best with ONLY the given models swapped
to their head-leak decoded slot (from reports/build/headleak2/slot_<name>.pt).
Keeps every other model = current best. One-variable, attributable. Login-safe.
"""
import os
import sys
import torch

import utils as U
from build_submission import validate_submission


def main():
    base = torch.load(U.work_path("outputs_final", "submission_current_best.pt"), weights_only=False)
    models = [a for a in sys.argv[1:] if a.startswith("model")]
    slotdir = U.work_path("reports", "build", "headleak2")
    cand = {k: base[k].float().clamp(0, 1).clone() for k in U.MODELS}
    for m in models:
        s = torch.load(os.path.join(slotdir, f"slot_{m}.pt"), weights_only=False)["images"].float().clamp(0, 1)
        cand[m] = s
        print(f"swapped {m} <- head-leak slot")
    errs = validate_submission(cand)
    if errs:
        print("INVALID:", errs[:3]); return
    tag = "_".join(m.replace("model", "m") for m in models)
    out = U.work_path("outputs_final", f"merge_hlcnn_{tag}.pt")
    torch.save({k: cand[k].to(torch.float32).contiguous() for k in U.MODELS}, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
