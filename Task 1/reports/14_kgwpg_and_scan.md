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

## 2b. Finale Verdict-Runde (`src/eval_kgwpg_final.py`, ~23:30) — alle Varianten, volles Gate

7-Punkte-Gate (val, valHA, valHB, revA, revB, train A→B, train B→A), fit-train clean:

| Config | val | valHA | valHB | revA | revB | trA→B | trB→A | ≥champ |
|---|---|---|---|---|---|---|---|---|
| champion | 0.4384 | 0.5311 | 0.3816 | 0.5135 | 0.3824 | 0.3314 | 0.3802 | — |
| binned E5 | 0.4427 | 0.5231 | 0.3387 | 0.5133 | 0.3380 | 0.3527 | 0.4072 | 3/7 |
| binned E7 | 0.4452 | 0.5219 | 0.3381 | 0.5118 | 0.3376 | 0.3453 | 0.4085 | 3/7 |
| **analytic B** (exakte Null, \|logLR\|≤δ) | **0.4253** | 0.5105 | 0.3431 | 0.4987 | 0.3408 | 0.3621 | 0.4097 | 2/7 |

Drei neue Befunde:
1. **Train- und Val-Splits widersprechen sich systematisch**: kgwpg gewinnt BEIDE
   Train-Halves deutlich (+0.014…+0.028), verliert BEIDE Val-Halves (−0.008…−0.044).
   Der Effekt transferiert innerhalb von Train, aber nicht nach Validation ⇒
   Distribution-Shift der pgreen-konditionalen Green-Raten, nicht bloß Rauschen.
2. **Per-Scheme-Kosten sichtbar**: KGW 0.189→0.278, aber gumbel 0.617→0.600 und
   textseal 0.615→0.589 — kgwpg-FPs heben den gepoolten Threshold, die starken
   Kanäle bezahlen. Netto +0.007, Gate-Anforderung (+0.01, keine gum/ts-Regression) verletzt.
3. **Analytische LR (Variante B) verliert sogar pooled** (−0.013): der Soft-Logit-Tilt
   modelliert das Self-Salt-KGW falsch (Caveat bestätigt); da \|logLR\|≤1.5 per
   Konstruktion, bestätigt das unabhängig das Capped-Ergebnis (Caps 16/8/4/2.5
   retten nichts, `eval_kgwpg_capped.py`).

FP-Tail (Top-150 Clean, full val): champion 77% Boundary-Bleed (≤10 Tok. an echtem
Span), binned 91–93%, pg<0.05-Anteil ≈ Base-Rate (0.43 vs ~0.46) ⇒ kein grobes
pgreen-Overfitting; der Tail ist Boundary-Bleed, kgwpg konzentriert ihn nur stärker.
Variante D (empirische Log-Odds pro Bin, train-only, geglättet) = strukturell identisch
mit Variante A im HMM (per-Bin-Kategorial mit eps) — mitgetestet. Variante E
(Scheme-Normalisierung) per Auftrag übersprungen, da A–D kein robustes Signal zeigen.

## 3. Konsequenz für die Gap-Analyse (0.3744 vs. SeeWeedZ 0.4688)

Brainstorm-Scorecard: #1 KGW-LR gemessen→Artefakt · #2/#4 nativ/Scan gemessen→tot ·
#3 Dedup-Semantik strukturell abgedeckt (v2=offizielles scoring_method, kgw_first=
ignore_repeated_ngrams, unigram is_repeated_ngram_2) · #5 Public-Shaping durch
Endspiel-Züge ausgeschlossen. **Alle sechs Hypothesen sind lokal erschöpft.** Der Gap
ist mit unserer Informationsbasis nicht erklärbar; verbleibender Kanal = Discord
(Multi-Team-Sprung-Koordination). Task-1-Eintrag 0.3744 (sub_010_lp public best,
logprob-Champion) bleibt final, sofern kein externer Hinweis auftaucht.
