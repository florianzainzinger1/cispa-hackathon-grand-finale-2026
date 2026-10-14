# Project Overview — Task 1: Text Watermark Localization

> **See also:** [02_strategy.md](02_strategy.md) — the agreed solution strategy (signals → windows → p-values → fusion → smoothing → calibration).

**CISPA European Championship in Trustworthy AI — Grand Finale (July 2nd, 2026)**
Supervisors: Adam Dziedzic & Franziska Boenisch (SprintML group, CISPA). Task developed by Maitri Shah and Louis Kerner.

This report is a read-only survey of the repository as of the initial inspection. No code has been written or modified.

---

## 1. Repository structure

Repository root: `C:\Users\masho\Documents\Hackathon\CISPA Final\CISPA-Hackathon`
Git: single branch `main`, one commit (`3dceefc init`). The working tree contents are currently untracked (`?? ./` in git status).

```
CISPA-Hackathon/
├── .env.example                                  # API_KEY="YOUR_API_KEY" placeholder
├── .gitignore                                    # ignores ".env" only
└── task1/
    ├── .claude/settings.local.json               # local Claude Code permission settings (tooling, not project)
    ├── Task 1 Text Watermark Localization.pdf    # ★ authoritative Task 1 specification (5 pages)
    ├── Tasks-Final.pdf                           # slide deck covering Tasks 1, 2, and 3 (31 slides)
    ├── Intro-final.pdf                           # hackathon logistics/intro deck (14 pages)
    ├── kirchenbauer23a.pdf                       # reference paper: KGW watermark
    ├── 2605.12456v2.pdf                          # reference paper: TextSeal watermark
    ├── 3976_Provable_Robust_Watermark.pdf        # reference paper: Unigram watermark
    ├── watermark.ppt                             # reference slides: Gumbel-Max watermark (Aaronson)
    ├── 2112.02918v2.pdf                          # reference paper for ANOTHER task (Task 3, gradients)
    └── mgi.pdf                                   # reference paper for ANOTHER task (Task 2, MGI)
```

There is **no source code, no dataset, no YAML config, no notebook, and no script anywhere in the repository yet.** The repo is documentation-only at this point.

---

## 2. Relevant files for Task 1

| File | Role |
|---|---|
| `task1/Task 1 Text Watermark Localization.pdf` | **Primary task spec**: goal, dataset schema, evaluation metric, submission format, leaderboard, references. |
| `task1/Tasks-Final.pdf` (pages 1–13) | Slide version of Task 1: red-green list intuition, worked example, scoring, data splits, submission format. |
| `task1/kirchenbauer23a.pdf` | "A Watermark for Large Language Models" (Kirchenbauer et al., ICML 2023) — the **KGW** scheme (red/green lists seeded by previous-token hash). |
| `task1/2605.12456v2.pdf` | "TextSeal: A Localized LLM Watermark for Provenance & Distillation Protection" (Sander et al., FAIR/Meta, arXiv 2026) — **TextSeal**, built on Gumbel-max sampling. |
| `task1/3976_Provable_Robust_Watermark.pdf` | "Provable Robust Watermarking for AI-Generated Text" (Zhao et al., ICLR 2024) — the **Unigram** watermark (fixed global green list). |
| `task1/watermark.ppt` | Scott Aaronson's talk slides "Watermarking GPT Outputs" — the **Gumbel-Max** scheme (verified via file metadata: author Scott Aaronson, PowerPoint, last saved Dec 2022). |
| `.env.example` | Placeholder for `API_KEY` — presumably the leaderboard submission API key referenced by `submission_template.py` in the spec. *(Assumption: this key corresponds to `YOUR_API_KEY_HERE` in the submission template.)* |
| `task1/Intro-final.pdf` | Logistics only (schedule, WiFi, Discord, prizes, jury). Useful for context, not for solving the task. |

### Task summary (from the spec)

- **Input:** documents = tokenized text (Qwen/Qwen2.5-7B-Instruct tokenizer). Each document mixes human-written text, clean LLM-generated text, and one or more **watermark-active spans** (possibly several schemes per document).
- **Output:** one continuous confidence score in `[0, 1]` **per token** (per entry of `token_ids`), higher = more likely watermarked.
- **Metric:** **TPR @ 0.1% FPR**, pooled over all tokens of the leaderboard subset. The 0.5 threshold is descriptive; the pooled ROC-based metric is what counts — score *ranking* matters, and false positives are punished very hard.
- **Key subtlety:** ground truth = "was the watermarking process switched ON when this token was generated", **not** "does this token pass a detector rule". Clean tokens can pass detector rules by chance; watermarked tokens can fail them.
- **Watermark schemes present:** TextSeal, Gumbel-Max, Unigram, KGW. Identifying the family is *not* required — only localization.
- **Non-watermarked text can be:** human-created, clean (non-watermark) generation, or text watermarked with *schemes other than the four above* (per slide 7 of `Tasks-Final.pdf`) — these must all be scored as class 0.
- **Important KGW note (from spec):** KGW greenlists were generated with `torch.randperm` on a **CUDA generator (Philox)**. Recomputing greenlists on CPU yields effectively random assignments and destroys ~⅓ of the KGW signal. Detector reproduction must run on GPU/CUDA.

### Submission format

`.jsonl`, one object per test document, exactly two keys:

```json
{"document_id": "1", "scores": [0.02, 0.04, 0.91, 0.87]}
```

Constraints: every test `document_id` exactly once, no extras/duplicates, scores 1-D, numeric, finite, in `[0,1]`, length exactly equal to that document's `token_ids` length. Invalid submissions are auto-rejected. Leaderboard: `http://<SUBMIT_SERVER>/leaderboard_page`; public split = 30% of test docs, private = 70% (split by hash of `document_id`, whole documents stay in one split); only the best score per team is kept.

---

## 3. Irrelevant files / files likely belonging to other tasks

| File | Why irrelevant to Task 1 |
|---|---|
| `task1/mgi.pdf` | "MGI: Member vs Generated Inference" (Zhao, Meintz, Xu, Boenisch, Dziedzic — CISPA). This is the reference paper for **Task 2** (fool an M/N/G classifier on images). |
| `task1/2112.02918v2.pdf` | "When the Curious Abandon Honesty: Federated Learning Is Not Private" (Boenisch et al.). Reference for **Task 3** (data reconstruction from gradients). |
| `task1/Tasks-Final.pdf` (pages 14–31) | Task 2 (MGI) and Task 3 (gradient reconstruction) slides — other tasks. |
| `task1/.claude/settings.local.json` | Local Claude Code tool permissions; not project material. |
| `task1/Intro-final.pdf` | Event logistics only. |

*(Note: `mgi.pdf` and `2112.02918v2.pdf` sit inside the `task1/` folder but clearly belong to Tasks 2 and 3.)*

---

## 4. Dataset files and schemas

**No dataset files are present in the repository.** The dataset must be downloaded from:

> `https://huggingface.co/datasets/SprintML/watermark_localization`

The spec also references a **`hackathon_setup.sh`** script ("creates your team folder, downloads the datasets and creates a per-task environment") — **not present in this repo**.

### Splits (from spec)

| Split | Documents | Labels |
|---|---|---|
| Training | 90 | yes |
| Validation | 90 | yes |
| Test | 1,320 | **no** (hidden) |
| Total | 1,500 | |

### Train/validation record schema

```json
{
  "document_id": "train_1",
  "text": "Paul's statement",
  "token_ids": [25300, 594, 5114],
  "token_pieces": ["Paul", "'s", " statement"],
  "labels": [0, 0, 1]
}
```

- `token_ids` — **authoritative** token sequence (Qwen/Qwen2.5-7B-Instruct tokenizer). Do **not** re-tokenize `text`; decode/re-encode is not guaranteed to round-trip.
- `token_ids`, `token_pieces`, `labels` are aligned, same length; position `i` describes one token.
- `labels`: 1 = generated while a watermarking process was active, 0 = otherwise.
- Tokenizer notation: `Ġword` = leading space, `Ċ` = newline; words may split into multiple pieces.
- **All** tokens require predictions, including chat-template markers and apparent user/assistant messages.

### Test record schema

Same minus `labels`:

```json
{
  "document_id": "1",
  "text": "The complete document text...",
  "token_ids": [25300, 594, 5114],
  "token_pieces": ["Paul", "'s", "Ġstatement"]
}
```

*(Assumption: the dataset ships as `.jsonl`/zip given the spec mentions "code to unzip and load the dataset" in `task_template.py`; exact file layout unknown until downloaded.)*

---

## 5. Config/YAML files and important parameters

**No YAML file is present in the repository**, but the spec promises one:

> "Participants are provided with a YAML file containing the exact tokenizer, watermark keys, and detector parameters used in the task. The YAML also specifies the official watermark repositories and the exact commits used…"

Expected YAML contents (from spec, exact values unknown until obtained):

- Tokenizer: `Qwen/Qwen2.5-7B-Instruct`
- Watermark **keys** for each scheme (TextSeal, Gumbel-Max, Unigram, KGW)
- **Detector parameters** per scheme (e.g., KGW context width/γ/δ, Unigram green fraction, Gumbel/TextSeal seeds — assumption based on the papers)
- **Repository URLs + exact commits** of the official watermark implementations to clone for detector code

Other config-adjacent files:

- `.env.example` → `API_KEY="YOUR_API_KEY"` (leaderboard submission key; real `.env` is gitignored).
- `task1/.claude/settings.local.json` → Claude Code permission allowlist only.

---

## 6. Existing code and what it does

**There is no code in the repository.** Zero `.py`, `.ipynb`, `.sh`, `.yaml`/`.yml` files exist (verified by full-tree file listing).

Code referenced by the spec but **not yet in the repo**:

| Referenced file | Purpose per spec |
|---|---|
| `hackathon_setup.sh` | Creates team folder, downloads datasets, creates per-task environment. |
| `task_template.py` | Starter: unzips/loads dataset, generates/validates/saves continuous predictions. |
| `submission_template.py` | Submits the `.jsonl` scores file to the evaluation API (replace `YOUR_API_KEY_HERE` and `/PATH/FILE.jsonl`). |
| Official watermark repos (pinned commits, listed in the YAML) | Detector implementations for the four schemes. |

---

## 7. Missing files / unclear assumptions

Missing artifacts (must be obtained before implementation):

1. **The dataset** — download from HF `SprintML/watermark_localization`.
2. **The YAML config** (tokenizer, keys, detector params, repo commits) — location not stated in the PDFs; presumably distributed via `hackathon_setup.sh`, the HF dataset repo, or Discord.
3. **`hackathon_setup.sh`**, **`task_template.py`**, **`submission_template.py`** — referenced but absent.
4. **The actual API key** — `.env.example` is a placeholder; the real key must be set in `.env` (gitignored).
5. **Watermark detector repositories** — to be cloned at the exact commits given in the YAML.

Assumptions made in this report (marked throughout):

- **[A1]** `.env.example`'s `API_KEY` is the leaderboard submission key used by `submission_template.py`.
- **[A2]** The dataset is distributed as zipped `.jsonl` (inferred from "code to unzip and load the dataset").
- **[A3]** `mgi.pdf` and `2112.02918v2.pdf` were placed in `task1/` for team reference only and are not needed for Task 1.
- **[A4]** Detector parameters in the YAML will follow the parameterizations of the four cited papers (γ/δ/context for KGW, etc.).
- **[A5]** A CUDA-capable GPU will be needed to faithfully reproduce KGW greenlists (spec's Philox note), and likely useful for running Qwen2.5-7B for perplexity/logit-based features.

## Open Questions

1. **Where exactly is the YAML file?** Not in this repo. Is it inside the HF dataset repo, delivered by `hackathon_setup.sh`, or posted on Discord?
2. **Where are `hackathon_setup.sh` / `task_template.py` / `submission_template.py`?** Same distribution question.
3. **What is the submission API endpoint?** The spec names the leaderboard page (`http://<SUBMIT_SERVER>/leaderboard_page`) but the submit endpoint is presumably inside `submission_template.py`.
4. **Is there a submission rate limit / cap on attempts?** Not stated in the available documents.
5. **Do we have CUDA hardware available?** Required for faithful KGW greenlist reproduction (Philox/CUDA `torch.randperm` note) — CPU-only recomputation loses ~⅓ of the KGW signal.
6. **Are the "other watermarks" (class 0, scheme not among the four) actually present in train/validation**, or only in test? The slides only say non-watermarked text *may* include "different watermarks than the above".
7. **Span statistics unknown** until the data is inspected: typical span lengths, number of spans per document, share of human vs clean-generated vs watermarked tokens, document length distribution.
8. **Does `text` include chat-template markers literally** (e.g., `<|im_start|>`)? The spec says chat-like markers are included in evaluation; exact form needs verification in the data.

---

## 8. Glossary of project-specific terms

| Term | Meaning |
|---|---|
| **Watermark localization** | Token-level prediction of *where* in a document a watermarking process was active (vs. document-level detection). |
| **Watermark-active region / span** | Contiguous run of tokens generated while watermark sampling was switched on. Ground-truth labels come from this generation-time state, not from detector output. |
| **KGW** | Kirchenbauer–Geiping–Wen et al. red/green-list watermark: the previous token's hash seeds an RNG that partitions the vocabulary into a "green list" (boosted/forced) and "red list"; detection = statistical test on green-token frequency. |
| **Green list / red list** | The pseudo-random vocabulary partition used by KGW-style schemes; watermarked text over-represents green tokens. |
| **Unigram watermark** | Zhao et al. (ICLR 2024) variant with one **fixed, global** green list (context-independent), giving provable robustness to edits. |
| **Gumbel-Max watermark** | Aaronson & Kirchner's scheme: pseudorandom Gumbel noise (seeded by key + preceding context) drives deterministic `argmax` sampling; detection correlates tokens with the pseudorandom values. |
| **TextSeal** | Meta/FAIR watermark built on Gumbel-max sampling, designed for **localized** detection (provenance & distillation protection). |
| **Philox / CUDA generator note** | KGW greenlists in this dataset were computed with `torch.randperm` on a CUDA Philox RNG; CPU recomputation produces different permutations, so detectors must reproduce them on GPU. |
| **TPR @ 0.1% FPR** | Evaluation metric: true-positive rate on watermarked tokens at the score threshold where only 0.1% of clean tokens are (wrongly) flagged; computed after pooling all tokens in the leaderboard subset. |
| **Public / private leaderboard** | Deterministic 30% / 70% split of test documents by hash of `document_id`; public score visible during the competition, private decides final ranking. |
| **`token_ids` / `token_pieces`** | Authoritative token sequence (Qwen2.5-7B-Instruct tokenizer) and its human-readable piece strings (`Ġ` = leading space, `Ċ` = newline). |
| **Clean text** | Class-0 text: human-written, non-watermarked model generation, or text watermarked by a scheme outside the four provided ones. |
| **MGI (Member vs Generated Inference)** | **Task 2** (not ours): perturb images so a classifier misclassifies Member/Non-member/Generated origin, at minimal MSE. |
| **Data Reconstruction from Gradients** | **Task 3** (not ours): honest-but-curious FL server reconstructs user photos from shared gradients; scored by SSIM. |
| **SprintML** | The CISPA research group (Secure, Private, Robust, INterpretable, Trustworthy ML) running these tasks; also the HF org hosting the dataset. |

---

## 9. UPDATE (2026-07-02, after cluster setup) — Open Questions resolved

Team setup on JURECA is complete. All previously missing artifacts now exist at
`/p/scratch/training2625/zainzinger1/1337/watermark_localization/` (owner: `zainzinger1`; team: syed1, eiter1, bibulatov1). The folder contains `train.jsonl`, `validation.jsonl`, `test.jsonl`, `watermark_config.yaml`, `task_template.py`, `submission_template.py`, `requirements.txt`, and a ready `.venv` (torch, transformers, accelerate, sentence-transformers, numpy, scipy, pandas).

### Resolved: the YAML (`watermark_config.yaml`)

- **Tokenizer:** `Qwen/Qwen2.5-7B-Instruct`, pinned revision `a09a35458c702b33eeacc393d103063234e8bc28`.
- **Repositories (pinned commits):**
  - TextSeal **and** Gumbel-Max: `github.com/facebookresearch/textseal` @ `788fe8bff5cf086f0881928ce9a81aa08c21dff1` — detector entry points `textseal.watermarking.config.WatermarkConfig`, `textseal.watermarking.detector.TextSealDetector`.
  - KGW: `github.com/jwkirchenbauer/lm-watermarking` @ `82922516930c02f8aa322765defdb5863d07a00e` — entry point `extended_watermark_processor.WatermarkDetector`.
  - Unigram: `XuandongZhao/Unigram-Watermark` @ `b96cdb4d52771e3cbd543a9d9aeeaec8d0790ca2`.
- **Watermark parameters/keys:**
  - `textseal`: key_a=947821031, key_b=1562881159, ngram=3, mixing_alpha=0.5, method=uniform, scoring_method=v1
  - `gumbelmax`: secret_key=2004683203, ngram=2, method=uniform, scoring_method=v1
  - `kgw`: gamma=0.25, delta=1.5, seeding_scheme=`ff-anchored_minhash_prf-4-True-1306382177`, select_green_tokens=true, ignore_repeated_ngrams=true
  - `unigram`: fraction=0.5, strength=1.0, watermark_key=1873092841
- **Hidden per disclosure section:** which family is active at each token, test labels, span boundaries, private state arrays.

### Resolved: submission

`submission_template.py` POSTs the `.jsonl` to `http://<SUBMIT_SERVER>:80/submit/30-watermark-localization` with header `X-API-Key` (key from `.env`). `task_template.py` provides JSONL load/validate/write helpers and a random-scores baseline; its `validate_predictions()` mirrors the server-side checks.

### Resolved: dataset statistics (computed on cluster)

| Split | Docs | Tokens | Doc length min/med/max | Watermarked tokens | Docs w/ watermark | Spans/doc |
|---|---|---|---|---|---|---|
| train | 90 | 95,138 | 538 / 1060 / 1528 | 44,946 (47.2%) | 88 of 90 | 0–6 |
| validation | 90 | 95,704 | 542 / 1040 / 1527 | 44,325 (46.3%) | 88 of 90 | 0–6 |
| test | 1,320 | 1,394,156 | 513 / 1063 / 1536 | hidden | hidden | hidden |

Notes: ~47% of labeled tokens are watermarked; 2 docs per labeled split contain no watermark at all; documents contain up to 6 separate watermark spans. Test doc IDs are plain numbers ("1", …), train/val IDs are "train_1"/"validation_1" style.

### Still open

- Submission rate limits are still unknown (not in the templates).
- Whether train/val contain "other-scheme" watermarks as class 0 remains unverified.
- The real API key still needs to be placed in `.env` / `submission_template.py`.
