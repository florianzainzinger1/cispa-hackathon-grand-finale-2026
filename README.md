# CISPA Hackathon Championship 2026: Grand Finale

Code and research notes of team **!!1337** at the Grand Finale of the CISPA European
Cybersecurity and AI Hackathon Championship (CISPA Helmholtz Center for Information
Security, July 2026). The team qualified as one of the three winning teams of the
[Vienna regional round](https://github.com/florianzainzinger1/cispa-hackathon-championship-vienna)
and finished **4th overall** in the finale.

The finale was a 24-hour competition on trustworthy machine learning. Tasks were
scored on hidden leaderboards and run on GPU nodes of the JURECA supercomputer
(Jülich Supercomputing Centre).

## Tasks and approaches

### Task 1: Text watermark localization ([`task1/`](task1))

Given LLM-generated documents that mix human text, clean model output and spans
watermarked with one of four schemes (KGW, Unigram, Gumbel-Max, TextSeal), predict a
per-token probability that the watermark was active. The metric is TPR at 0.1% FPR,
pooled over all tokens, so false positives are extremely expensive.

- Per-token detector signals for all four schemes, plus Qwen2.5 entropy and log-probability
  features that capture how much "room" the sampler had to embed a watermark
- A 5-state hidden Markov model with factorised emissions conditioned on entropy and
  log-probability, later replaced by an explicit-duration HSMM that learns realistic
  span lengths and removes false positives from bridged gaps
- A pre-registered ship gate (validation halves and reverse splits must all improve)
  to avoid overfitting to the 30% public leaderboard split

Result: **public score 0.4240, rank #2** on the Task 1 leaderboard at the end of the event.
The full experiment log is in [`task1/submissions/submissions_log.md`](task1/submissions/submissions_log.md)
and [`task1/reports/`](task1/reports).

### Task 3: Federated-learning audit via gradient inversion ([`task3/`](task3))

Reconstruct 1,536 private 64×64 face images (12 models × 128 images) from
batch-averaged federated-learning gradients. Score: Hungarian-matched mean SSIM.

- Analytic first-layer readout (`x = ∂L/∂W_i / ∂L/∂b_i`) per model family (MLP, CNN, ViT)
- Low-rank SVD denoising of the first-layer gradients, with a per-model noise-floor
  cut-off, to separate the superimposed images in a batch of 128
- Convolution inversion for CNNs, selection of diverse candidates, and adaptive
  post-processing
- Slurm job scripts and a local → GitHub → JURECA deployment workflow ([`infra/`](infra))

Best team submission: **SSIM 0.3888** on the public split. The method notes are in
[`task3/APPROACH.md`](task3/APPROACH.md) and
[`task3/reconstruction/APPROACHES_LEDGER.md`](task3/reconstruction/APPROACHES_LEDGER.md).

## Repository layout

```
task1/            Task 1: detectors, HMM/HSMM decoders, evaluation gates, reports
task3/            Task 3: gradient-inversion pipeline, merged pipeline, reports
infra/            Slurm job + deployment script for the JURECA cluster
```

Datasets, model checkpoints, generated caches, submission files and the organisers'
task material are **not included**: they belong to the organisers and were only
licensed for the event. The code is published for documentation; it cannot be run
end-to-end without that data.

## Team

| Member | Main focus |
|---|---|
| Florian Zainzinger ([@florianzainzinger1](https://github.com/florianzainzinger1)) | Task 3 gradient-inversion pipeline, cluster workflow |
| Mashood Syed ([@mashoodsyed21](https://github.com/mashoodsyed21)) | Task 1 watermark localization |
| Albin | Task 3 merged pipeline, setup |
| Mansur | Team member |

Tooling note: the team used AI coding assistants (Claude Code) heavily during the event.
The reports in this repository are partly written by these agents as working notes.

## Context

- Organiser: [CISPA Helmholtz Center for Information Security](https://cispa.de), SprintML group
- Vienna qualifier (one of three winning teams): [cispa-hackathon-championship-vienna](https://github.com/florianzainzinger1/cispa-hackathon-championship-vienna)
