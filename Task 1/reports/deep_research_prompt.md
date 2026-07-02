# Deep Research Prompt — LLM Watermark Localization (für Gemini/ChatGPT Deep Research)

> Copy den Block unterhalb der Linie 1:1 in Gemini Deep Research (oder ChatGPT Deep Research). Er ist selbst-enthaltend.

---

**Role & Goal.** You are a research assistant for a machine-learning competition on **text watermark localization**. The task: for every token in a document, predict whether it was generated while an LLM watermarking process was active. Documents mix human text, clean model text, and watermarked spans from possibly different schemes. The evaluation metric is **True Positive Rate at 0.1% False Positive Rate (TPR@0.1%FPR)**, pooled across all tokens — so the decisive challenge is the extreme low-false-positive tail of the score ranking.

**Watermark schemes present in the data (all four, keys known):**
- **KGW** (Kirchenbauer et al., ICML 2023, "A Watermark for Large Language Models") — context-hashed red/green vocabulary split, gamma=0.25, delta=1.5, seeding scheme `anchored_minhash_prf`, context width 4, self-salted.
- **Unigram** (Zhao et al., ICLR 2024, "Provable Robust Watermarking for AI-Generated Text") — a single fixed global green list, fraction 0.5.
- **Gumbel-Max / exponential-minimum sampling** (Aaronson & Kirchner 2023) — distortion-free: pick token = argmax_x u_x^(1/p_x) with pseudorandom u_x; detection scores −log(1−u) on observed tokens.
- **TextSeal** (Meta/FAIR, arXiv:2605.12456, repo `facebookresearch/textseal`) — dual-key, entropy-weighted, localized detection; built on Gumbel-max sampling.

**Generation model / tokenizer:** Qwen/Qwen2.5-7B-Instruct (tokenizer pinned; generation model presumed the same).

**Our current approach (already strong):** exact reconstruction of each scheme's per-token PRF signal using the known keys; a 5-state HMM (clean + one state per scheme) over span structure; emission distributions conditioned on the token's Qwen entropy and log-probability, normalized within each conditioning level. Windowing/segmentation/change-point/HMM-tuning is exhausted for us — we need **new information sources**, not better smoothing.

**Answer these four questions with concrete, sourced, actionable findings** (cite papers, official repos, exact code paths, parameter defaults, and formulas wherever possible):

**1. Generation parameters.** What sampling temperature and top-p are used to *generate* Gumbel-Max and TextSeal watermarked text in the standard/official setups? Specifically: what are the defaults in the official TextSeal repo (`facebookresearch/textseal`, look at generation config / `ProcessingConfig` / demo scripts) and in Aaronson-style Gumbel-max implementations? Does the sampling temperature change the detection statistic or its calibration, and how does one correct a detector for a known generation temperature? (We suspect T=0.9, top-p=0.95 — verify or refute with the actual repo defaults.)

**2. TextSeal localized detection.** Explain TextSeal's own *localized detection* algorithm precisely: the geometric-cover / multi-region search, the per-segment test statistic, the entropy weighting, and the dual-key fusion. What is the core insight versus a single global test? Cite the paper (arXiv:2605.12456) sections and the repo functions (e.g. `localized_detect`, `_geometric_cover_search`, `TextSealDetector`). What is transferable to a **token-level** localization model?

**3. Optimal detection statistic per scheme.** What is the theoretically most powerful per-token or per-window test statistic for each scheme (KGW, Unigram, Gumbel-Max, TextSeal)? In particular: are there **likelihood-ratio / Neyman–Pearson-optimal** or score-based detectors sharper than the standard green-count z-test (for KGW/Unigram) and the Gamma test on Σ−log(1−u) (for Gumbel)? Keywords: optimal watermark detection, Neyman-Pearson watermark, log-likelihood-ratio watermark detection, entropy-weighted / "spike entropy" detection, robust/adaptive thresholding. Give the exact statistic and its null/alternative distributions where available.

**4. State-of-the-art watermark localization (2024–2026).** What methods exist for token-level or span-level localization of LLM watermarks in mixed documents? Cover at least: **WaterSeeker**, and any work on segment-level watermark detection, watermark change-point detection, "adaptive/localized" watermark detection, and watermarking under mixed/edited text. For each, state the core method and — critically — whether it introduces a **new information source** beyond window statistics + smoothing, or is just another segmenter. Prefer methods with public code.

**Output format:** For each of the four questions, give (a) the direct answer, (b) the strongest supporting source with a link, (c) a confidence level, and (d) a one-line "how we could implement this in a few hours" note. End with a ranked list of the top 3 most promising *new* ideas to try, most actionable first. Avoid general surveys — be specific, quantitative, and implementation-oriented.
