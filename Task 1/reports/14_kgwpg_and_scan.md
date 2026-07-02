# Report 14 — KGW exact LR (kgwpg) & Scan-Statistik: beide gemessen, beide geschlossen

**Jul 2, ~23:15.** Brainstorm-Hypothesen des Users systematisch abgearbeitet. Scripts:
`src/scan_localize.py`, `src/eval_kgwpg.py` / `eval_kgwpg2.py` / `eval_kgwpg3.py`.
Cluster: `output/extract_topk.py` (top-48 Qwen-Kandidaten, Masse-Coverage 0.985),
`output/extract_kgw_pgreen.py` (bit-exakte Greenlist-Masse via Repo-`_get_greenlist_ids`).

## 1. KGW exact LR / kgw_pgreen — der eine nie gemessene Hebel: JETZT GEMESSEN

Extraktion validiert durch Kalibrierung auf Clean-Tokens (pred vs. actual green rate):
pg[0,0.1): 0.018/0.026 · pg[0.2,0.3): **0.247/0.247** · pg[0.5,1]: 0.831/0.809. Verteilung
stark bimodal (p5=0.000, p95=0.998, mean 0.255 ≈ γ) — die Information existiert real.

**Pooled val (fit train, trans 0.03/80):**

| Config | T@0.1% | KGW-Recall |
|---|---|---|
| Champion (kgw=entropy) | 0.4384 | 0.189 |
| kgwpg [.02,.10,.25,.45,.65] | 0.4427 | 0.273 |
| kgwpg [.01,.04,.10,.20,.35,.55,.80] | **0.4452** | 0.278 |
| kgwpg×entropy joint | ≤0.4405 | — |

KGW-Recall +44% relativ, aber nur +0.007 pooled — Report-13-Ceiling bestätigt
(globaler Threshold vom Gumbel/TS-FP-Tail gedeckelt).

**Challenger-Gate (Split-Half + Reverse-Split): DURCHGEFALLEN — 4/4 Niederlagen.**

| Eval | Champion | kgwpg E5 | kgwpg E7 |
|---|---|---|---|
| halfA fit-train | **0.5311** | 0.5231 | 0.5219 |
| halfB fit-train | **0.3816** | 0.3387 | 0.3381 |
| reverse evalA | **0.5135** | 0.5133 | 0.5118 |
| reverse evalB | **0.3824** | 0.3380 | 0.3376 |

Der pooled Gewinn ist ein Threshold-Artefakt: die extremen Low-pgreen-Level verdünnen
die Clean-Null (P(green|clean, pg<0.02)≈0.02 statt 0.25) ⇒ seltene Zufalls-Greens werden
zu massiven per-Token-LRs ⇒ heavy-tailed FPs, die bei per-Half-Thresholds (~47 FP-Budget)
das gesamte 0.1%-Budget fressen. Half B kostet es −0.044. Gleiches Artefakt-Muster wie
sub_007 (rank: lokal +, public −0.016) — nur mit noch schlechterer Evidenz (rank gewann
wenigstens die Splits). **Nicht shippen. Test-Extraktion (~6 GPU-h) nicht gestartet.**
Infra bleibt: `--cond pgexact` in predict_hmm.py + kgwpg-Caches train/val, topk train/val/test.

## 2. Scan-Statistik / natives TextSeal `localized_detect` — geschlossen

Repo-Fund: `textseal/watermarking/detector.py` enthält `localized_detect()` mit
`_geometric_cover_search` (dyadische Fenster 2^p, Stride len/2), Gamma-Region-p-Wert,
Bonferroni (`_count_tests`), `_boundary_smoother` — d.h. Brainstorm-Hypothesen #2 UND #4
in einer Funktion, aus derselben Datei wie unser v2-Fund (scoring_method, Z. 358).

Port in `scan_localize.py`, in stärkster Form (Evidenz = unsere HMM-Emissions-LLR
aller 4 Schemes inkl. Konditionierung, Null-Momente aus Train-Clean):

| Variante | T@0.1% |
|---|---|
| Champion HMM | **0.4384** |
| Scan alle Schemes (min_len 16/24/32/64) | 0.0104 |
| Fusion max/mean/0.8-blend mit Champion | 0.15–0.25 |
| Scan kgw-only + Fusion | ≤0.2521 |

Grund: Der Scan gibt fenster-uniforme Scores — bei GEPOOLTER Token-Metrik kann er
innerhalb eines Dokuments nicht differenzieren; `localized_detect` ist ein
Dokument-Level-Detektor (p-Wert „gibt es eine WM-Region?"), kein Token-Ranker.
Span-Level-Aggregation macht unser Forward-Backward bereits optimal weich.

## 3. Konsequenz für die Gap-Analyse (0.3744 vs. SeeWeedZ 0.4688)

Brainstorm-Scorecard: #1 KGW-LR gemessen→Artefakt · #2/#4 nativ/Scan gemessen→tot ·
#3 Dedup-Semantik strukturell abgedeckt (v2=offizielles scoring_method, kgw_first=
ignore_repeated_ngrams, unigram is_repeated_ngram_2) · #5 Public-Shaping durch
Endspiel-Züge ausgeschlossen. **Alle sechs Hypothesen sind lokal erschöpft.** Der Gap
ist mit unserer Informationsbasis nicht erklärbar; verbleibender Kanal = Discord
(Multi-Team-Sprung-Koordination). Task-1-Eintrag 0.3744 (sub_010_lp public best,
logprob-Champion) bleibt final, sofern kein externer Hinweis auftaucht.
