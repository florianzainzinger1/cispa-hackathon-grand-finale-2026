"""Agent B — integrator. Assemble outputs_final/submission_current_best.pt from the
best available source PER MODEL, keeping an always-valid best submission.

Sources considered per model (whichever exist on disk):
  current_best  outputs_final/submission_current_best.pt   (incumbent — the file we own)
  c3            outputs_v3/submission_v3.pt                 (frozen 0.3325 baseline)
  a1            outputs_a1/submission_a1.pt                 (built, unscored MLP-SVD + analytic-ViT)
  retr          outputs_retr/retrieved_{model}.pt           (Agent R — verbatim retrieval)
  trap          outputs_trap/trap_{model}.pt                (Agent T — exact analytic)
  bss           outputs_bss/bss_{model}.pt                  (Agent S — source separation)
  gp            outputs_gp/gp_{model}.pt                    (Agent G — generative prior)
  +pp           postprocess_model(incumbent)                (L5 — grain-guarded self-improve)

Selection rule (GT-free, avoids the c5 mistake where grainier-but-higher-capture LOST):
  * a candidate replaces the incumbent only if it PARETO-DOMINATES on the proxy
    (capture >=, grain <=, diversity >=, strictly better on one) — see proxy_score.dominates.
  * a verified retrieval source gets an override: adopted if capture >= incumbent
    and grain not worse (verbatim originals are both faithful AND clean).
  * otherwise the incumbent is kept. Nothing can silently degrade the shipped best.

Each chosen non-curated source is filled to exactly 128 diverse real faces (bank fill,
NEVER noise) using the v3 selection engine. Output is strictly validated and a manifest
records which model came from which source + the deciding metrics.

Never overwrites outputs/submission.pt, outputs_v3, or the _c*/_a1 artifacts.
Login-node safe.
"""
import os
import sys
import json
import torch

import utils as U
from build_submission import validate_submission
import proxy_score as PS
import merge_postproc as PP
from v3_select import (quality_and_stats, is_real, feat_vectors,
                       greedy_dedup, farthest_point_fill)


# ---------------------------------------------------------------------------
# bank + short-set completion
# ---------------------------------------------------------------------------
def load_bank():
    for p in (U.work_path("outputs_v4", "face_bank_v4.pt"),
              U.work_path("outputs_v3", "face_bank_v3.pt"),
              U.work_path("outputs", "face_bank.pt")):
        if os.path.exists(p):
            return torch.load(p, weights_only=False)["images"].float().clamp(0, 1)
    return None


def ensure_128(imgs, bank, n=128, tv_cut=0.22, dedup_eps=0.03):
    """Turn an arbitrary candidate pool into exactly n diverse real faces:
    drop noise, greedy-dedup by quality, fill shortfall from the bank (never noise).
    Returns (imgs_128, provenance dict)."""
    imgs = imgs.float().clamp(0, 1)
    q, st = quality_and_stats(imgs)
    real = is_real(st, tv_cut=tv_cut)
    feats = feat_vectors(imgs)
    idx_real = torch.nonzero(real, as_tuple=False).squeeze(1)
    reps, have_feats = [], []
    if idx_real.numel() > 0:
        sub_imgs, sub_q, sub_feats = imgs[idx_real], q[idx_real], feats[idx_real]
        chosen_local, chosen_feats = greedy_dedup(sub_imgs, sub_q, sub_feats,
                                                  n=n, dedup_eps=dedup_eps)
        reps = [sub_imgs[li] for li in chosen_local]
        have_feats = chosen_feats
    prov = {"from_source": len(reps)}
    n_needed = n - len(reps)
    if n_needed > 0 and bank is not None:
        bank_feats = feat_vectors(bank)
        fill = farthest_point_fill(bank, bank_feats, have_feats, n_needed)
        reps += [bank[j] for j in fill]
        prov["from_bank"] = len(fill)
    while len(reps) < n:                                   # last resort, never noise
        reps.append(reps[len(reps) % max(1, len(reps))] if reps else torch.zeros(3, 64, 64))
    prov["total"] = n
    return torch.stack(reps[:n]).clamp(0, 1).float(), prov


# ---------------------------------------------------------------------------
# source discovery
# ---------------------------------------------------------------------------
def _extract(obj):
    """Pull a (N,3,64,64) tensor out of a submission dict / {'images':...} / bare."""
    if torch.is_tensor(obj):
        t = obj
    elif isinstance(obj, dict):
        if "images" in obj:
            t = obj["images"]
        else:
            return None
    else:
        return None
    t = t.float()
    return t if t.ndim == 4 and t.shape[1:] == (3, 64, 64) else None


def gather_sources(name, incumbent_sub, bank):
    """Return {src_name: (imgs_128, curated_bool, is_retrieval_bool)} for one model."""
    out = {}
    # incumbent (always present)
    inc = _extract(incumbent_sub.get(name))
    if inc is not None and inc.shape[0] == 128:
        out["current_best"] = (inc, True, False)

    # curated per-model-in-submission sources
    for tag, path in [("c3", U.work_path("outputs_v3", "submission_v3.pt")),
                      ("a1", U.work_path("outputs_a1", "submission_a1.pt"))]:
        if os.path.exists(path):
            sub = torch.load(path, weights_only=False)
            t = _extract(sub.get(name))
            if t is not None and t.shape[0] == 128:
                out[tag] = (t, True, False)

    # per-model agent outputs (may be <128 -> completed via bank)
    for tag, fname, is_retr in [
        ("retr", f"retrieved_{name}.pt", True),
        ("trap", f"trap_{name}.pt", False),
        ("bss",  f"bss_{name}.pt", False),
        ("gp",   f"gp_{name}.pt", False),
    ]:
        d = {"retr": "outputs_retr", "trap": "outputs_trap",
             "bss": "outputs_bss", "gp": "outputs_gp"}[tag]
        path = U.work_path(d, fname)
        if os.path.exists(path):
            t = _extract(torch.load(path, weights_only=False))
            if t is not None and t.shape[0] > 0:
                imgs128, _ = ensure_128(t, bank) if t.shape[0] != 128 else (t, None)
                out[tag] = (imgs128, False, is_retr)

    # L5 post-processed variants (grain-guarded). Agent recons (trap/bss/gp) tend to
    # be higher-capture / more-diverse but GRAINIER than the clean bank, so the raw
    # grain gate rejects them. Degraining them (force denoise+moment) can unlock a
    # genuine Pareto win (capture>=, diversity>=, grain now <=) — unlike c5 which
    # RAISED grain. Reference colours from the incumbent's clean MLPs.
    rm, rs = PP.reference_from_clean_models(incumbent_sub)
    if "current_best" in out:
        pp = PP.postprocess_model(name, out["current_best"][0], ref_mean=rm, ref_std=rs)
        out["current_best+pp"] = (pp, True, False)
    force_pp = {name: dict(denoise=True, moment=True)}
    for tag in ("retr", "trap", "bss", "gp"):
        if tag in out:
            imgs, curated, is_retr = out[tag]
            pp = PP.postprocess_model(name, imgs, ref_mean=rm, ref_std=rs, config=force_pp)
            out[tag + "+pp"] = (pp, curated, is_retr)
    return out


# ---------------------------------------------------------------------------
# per-model decision
# ---------------------------------------------------------------------------
def decide(name, sources, mode="safe"):
    """Pick the winning source for one model. Returns (winner_tag, imgs, decision).
    mode='safe'  -> strict Pareto only (never risks a regression; guaranteed >= c3
                    on every proxy axis). This is what current_best uses.
    mode='aggr'  -> also accepts a candidate that is clearly higher CAPTURE and much
                    more DIVERSE at a bounded grain cost (Hungarian one-to-one SSIM
                    rewards coverage; a big diversity gain outweighs mild grain). This
                    is an A/B-test variant, NEVER written to current_best."""
    inc_tag = "current_best" if "current_best" in sources else next(iter(sources))
    # cheap metrics (no grad backward) for selection
    metrics = {tag: PS.score_images(name, imgs, want_gradcos=False)
               for tag, (imgs, _, _) in sources.items()}
    inc_m = metrics[inc_tag]
    winner, reason = inc_tag, "kept incumbent (no source Pareto-dominated it)"

    # retrieval override first
    for tag, (imgs, _, is_retr) in sources.items():
        if not is_retr:
            continue
        m = metrics[tag]
        cap_ok = (m["capture"] or -1) >= (inc_m["capture"] or -1) - 1e-3
        grain_ok = m["mean_tv"] <= inc_m["mean_tv"] + 5e-3
        if cap_ok and grain_ok:
            winner, reason = tag, "verified retrieval override (capture>=inc, grain not worse)"
            break

    # else best Pareto-dominator by proxy_scalar
    if winner == inc_tag:
        dominators = [tag for tag in sources
                      if tag != inc_tag and PS.dominates(metrics[tag], inc_m)]
        if dominators:
            winner = max(dominators, key=lambda t: metrics[t]["proxy_scalar"])
            reason = f"Pareto-dominates incumbent (scalar {metrics[winner]['proxy_scalar']} " \
                     f"vs {inc_m['proxy_scalar']})"

    # aggressive diversity-priority fallback (A/B variant only). Compares against the
    # CURRENT strict winner (not just the incumbent) so a marginal +pp Pareto win does
    # not block a much-more-diverse candidate. Hungarian one-to-one SSIM rewards
    # coverage, so a big diversity gain at bounded grain cost is a bet worth testing.
    if mode == "aggr":
        wm = metrics[winner]
        wc = wm["capture"] if wm["capture"] is not None else -1
        cands = []
        for tag, m in metrics.items():
            if tag == winner:
                continue
            mc = m["capture"] if m["capture"] is not None else -1
            cap_ok = mc >= wc - 5e-3
            div_win = m["mean_nn_dist"] >= wm["mean_nn_dist"] + 0.3
            grain_ok = m["mean_tv"] <= wm["mean_tv"] + 0.05
            if cap_ok and div_win and grain_ok and m["proxy_scalar"] > wm["proxy_scalar"]:
                cands.append(tag)
        if cands:
            new = max(cands, key=lambda t: metrics[t]["proxy_scalar"])
            reason = f"AGGR diversity-priority (nn {metrics[new]['mean_nn_dist']:.2f} " \
                     f"vs {wm['mean_nn_dist']:.2f}, cap {metrics[new]['capture']}, " \
                     f"grain +{metrics[new]['mean_tv']-wm['mean_tv']:.3f}) over {winner}"
            winner = new

    decision = {
        "winner": winner, "incumbent": inc_tag, "reason": reason,
        "candidates": {t: {k: metrics[t][k] for k in
                       ("capture", "mean_tv", "mean_nn_dist", "n_distinct", "proxy_scalar")}
                       for t in sources},
    }
    return winner, sources[winner][0], decision


# ---------------------------------------------------------------------------
# assemble
# ---------------------------------------------------------------------------
def main():
    # ALWAYS seed the incumbent from the FROZEN c3 baseline (the only source with a
    # real leaderboard score, 0.3325). Building from c3 + all on-disk agent sources
    # each run is idempotent; seeding from the evolving current_best would double-
    # postprocess (denoise-on-denoise) and creep. current_best.pt is OUTPUT, not input.
    c3 = U.work_path("outputs_v3", "submission_v3.pt")
    incumbent_sub = torch.load(c3, weights_only=False)
    print("[seed] incumbent = frozen c3 (outputs_v3/submission_v3.pt)")

    mode = os.environ.get("MERGE_MODE", "safe")
    outname = os.environ.get("MERGE_OUT", "submission_current_best.pt")
    # SAFETY: aggressive/A-B variants must NEVER clobber the canonical best.
    if mode != "safe" and outname == "submission_current_best.pt":
        outname = "submission_aggressive.pt"
    print(f"[mode] {mode}  -> outputs_final/{outname}")

    bank = load_bank()
    print(f"bank: {'loaded' if bank is not None else 'MISSING'}")

    sub, manifest = {}, {}
    for name in U.MODELS:
        sources = gather_sources(name, incumbent_sub, bank)
        if not sources:
            # keep incumbent verbatim if somehow no source built
            sub[name] = _extract(incumbent_sub[name]).float().clamp(0, 1)
            manifest[name] = {"winner": "current_best(verbatim)", "reason": "no sources"}
            continue
        winner, imgs, decision = decide(name, sources, mode=mode)
        sub[name] = imgs.float().clamp(0, 1).contiguous()
        manifest[name] = decision
        print(f"{name:8} -> {winner:16} : {decision['reason']}")

    # strict validation (organizer-equivalent)
    errs = validate_submission(sub)
    if errs:
        print("VALIDATION ERRORS:")
        for e in errs:
            print("  -", e)
        for k in sub:
            sub[k] = sub[k].clamp(0, 1).to(torch.float32)
        if validate_submission(sub):
            print("STILL INVALID — not writing.")
            sys.exit(1)

    out = U.work_path("outputs_final", outname)
    U.ensure_dir(os.path.dirname(out))
    torch.save({k: v.to(torch.float32).contiguous() for k, v in sub.items()}, out)

    # grad-cosine on the FINAL winners for the report (secondary signal)
    for name in U.MODELS:
        m = PS.score_images(name, sub[name], want_gradcos=True)
        manifest[name]["final_metrics"] = {
            k: m[k] for k in ("family", "capture", "grad_cos_first", "grad_cos_full",
                              "mean_tv", "n_distinct", "proxy_scalar")}

    U.save_json(manifest, U.work_path("reports", "build", f"build_manifest_{mode}.json"))
    preview = torch.cat([sub[m][:12] for m in U.MODELS], dim=0)
    U.save_image_grid(preview, U.work_path("outputs_final", f"{outname}.preview.png"), ncol=12)
    print("\nsaved:", out)
    print("manifest:", U.work_path("reports", "build", "build_manifest.json"))
    n_swapped = sum(1 for n in U.MODELS
                    if manifest[n].get("winner", "").split("(")[0] not in ("current_best",))
    print(f"models taken from a NON-incumbent source: {n_swapped}/12")


if __name__ == "__main__":
    main()
