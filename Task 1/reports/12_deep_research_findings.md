# Deep-Research-Ergebnisse — verwertbare Hebel

Interne Deep-Research (105 Agenten, 23 Quellen, 25 Claims adversarial verifiziert: 18 bestätigt, 7 widerlegt). Fokus: NEUE Informationsquellen statt Glättung.

## Der Haupttreffer: NP-optimaler Per-Token-Score (in Umsetzung)

**Unser `−log(1−u)`-Score (Aaronson) ist beweisbar suboptimal.** Für distortion-free Schemes (Gumbel-Max, TextSeal) ist der gewählte-Token-Pivot r=u unter H0 Uniform(0,1), unter H1 mit wahrer NTP-Verteilung P hat er die geschlossene Dichte

    f1(r) = Σ_w r^(1/P_w − 1)

⇒ Neyman-Pearson-optimaler Per-Token-Score = `log f1(r) = logsumexp_w[(1/P_w − 1)·log r]`.

- **Neue Informationsquelle**: nutzt die volle Verteilungs-*Form* P (aus Qwen), nicht nur die Skalare Entropie/Logprob. Wir haben P (Qwen) + u (gecacht) → exakt berechenbar.
- Quellen: arXiv 2404.01245 (Annals of Statistics, Thm 3.2/3.3), 2411.13868v2, 2505.08878 (ICLR 2025). NP-Optimalität bestätigt 3-0; geschlossene CDF bestätigt 3-0.
- **Scope-Caveat (verifiziert)**: gilt NUR für Gumbel/TextSeal. KGW/Unigram sind "biased" (NTP verändert), kein sauberer Pivot → dort bleibt der z-Test. Umsetzung: nplr als neuer Emissions-Observable für gumbel/textseal.

## GoF / Tr-GoF (Gemini-Recherche #1) — GETESTET, GESCHLOSSEN

Gemini's unabhängige Deep-Research rankte Tr-GoF als #1. Getestet in stärkster Form:
- High-u-Konzentration (Indikator u>0.9/0.95/0.8, Binomial-Tail): standalone 0.010–0.013.
- Echte **Higher-Criticism** über die volle CDF der u-Werte im Fenster (16/32): standalone **0.009–0.012**.

Beide weit unter unserer Fenster-Evidenz (0.05) und dem Champion (0.4255). **Warum (jetzt dreifach belegt):** GoF/Tr-GoF schlägt in der Literatur **Summen-Detektoren** auf ganzen Dokumenten. Unser HMM ist **Per-Token-Likelihood-Modellierung** — bei bedingt unabhängigen Watermark-Signalen ist das bereits die suffiziente Statistik. GoF fügt nichts hinzu, was das Per-Token-HMM nicht schon nutzt. Dasselbe Muster wie beim NP-LR: Beide „optimalen" Ansätze der Literatur adressieren ein Problem (Summen-Dilution), das unser HMM konstruktionsbedingt nicht hat.

## Zweitbefund: Goodness-of-Fit statt Summe (Reserve)

Statt `Σ −log(1−u)` über ein Fenster: GoF-Test (Anderson-Darling, Chi², Neyman-smooth) der {u_t} gegen Uniform(0,1). Nutzt die *ganze empirische CDF*, nicht nur die Summe. arXiv 2510.03944 (NeurIPS 2025 Spotlight): 2.9% vs 15.1% Type-II bei T=0.3. Theorie 2411.13868: additive Statistiken haben harte Grenze q+p=1/2, LRT/GoF erreichen q+2p=1. **Bestätigt 3-0.** Für uns Reserve — schwerer ins Per-Token-HMM zu integrieren als der LR-Score. Nur Gumbel/TextSeal (KGW ausgenommen).

## Bestätigte Detail-Verfeinerungen

- **TextSeal-Entropie-Gewichtung** (Paper Eq. 4): `w_ent = 0.1 + 0.9·(H−H_min)/(H_max−H_min)`, bestes Variant sogar *konkav* (sqrt). Validiert unsere Entropie-Konditionierung; konkave Transform als A/B-Test offen.
- **Cheap Fallback** (2404.01245 Thm 3.3, Crossover Δ*≈0.178 ↔ top-prob 0.82): pro Token — top-prob>0.82 → `−log(1−u)`, sonst `log(u)`. Falls voller LR zu teuer.
- **TextSeal = Dual-Key** mit α≈0.1 (nicht 0.5!): `s = (1−α)·s_A + α·s_B`. Wir behandeln beide Keys 50/50 — Prüfen ob α=0.1 besser.

## Widerlegt (nicht verfolgen)

- Generierungs-Temperatur als fester Default (0-3) — deckt sich mit unserem eigenen T-Experiment (Rauschen). Repo gibt nur Ranges T=0.7–1.0.
- Temperatur beeinflusst Detektierbarkeit monoton (0-3).
- WaterSeeker nutzt KEINE neue Informationsquelle — nur Fenster+Glättung (bestätigt: für uns wertlos).
- TextSeals exakter Localized-Detection-Algorithmus (geometric cover + Bonferroni) — Claim widerlegt 1-2, bleibt unverifiziert.

## Priorität

1. **NP-LR-Score** (läuft) — der eine echte neue Hebel, theoretisch optimal.
2. GoF-Fenster-Feature — falls #1 zündet und Zeit bleibt.
3. Konkave Entropie / α=0.1 Dual-Key — Feinschliff.

## ERGEBNIS NP-LR (implementiert, getestet) — kein Gewinn

Voll implementiert (Cluster-Job: volle Qwen-Verteilung P pro Position, `score = logsumexp_w[(1/P_w−1)·log u]`, GPU, für Gumbel + beide TextSeal-Keys; marginaler Dual-Key-LR = logmeanexp). Sanity: Signal trennt sauber (nplr_gumbel clean −0.73 vs wm −0.43). **Aber als Ersatz-Observable schlägt es den Champion nicht:**

| Variante | val @0.1%FPR |
|---|---|
| Baseline (Aaronson −log(1−u) + logprob-cond) | **0.4255** |
| NP-LR (10/14/20 bins) + logprob-cond | 0.371 / 0.370 / 0.363 |
| NP-LR + no-cond | 0.362 |
| NP-LR + entropy-cond | 0.362 |

**Warum die Theorie hier nicht durchschlägt (die eigentliche Erkenntnis):**
1. Per-Token-Optimalität ≠ Optimalität der span-gepoolten Metrik — das HMM aggregiert, und der per-Token-LR macht das Aggregat nicht optimal.
2. Unsere **Logprob-Konditionierung approximiert den NP-LR bereits nichtparametrisch** (Score moduliert durch P-Form) — und ist robust gegen die unbekannte Generierungs-Temperatur/top-p, während der parametrische NP-LR bei T=1 gegen die falsche Alternative kalibriert ist (Research-Caveat bestätigt sich empirisch).
3. NP-LR komprimiert (u, P-Form) in einen Skalar; unser (Aaronson-Score × Logprob-Level) behält den 2D-Produktraum, den die Emission reicher nutzt.

**Bedeutung:** Das ist ein wertvolles Negativergebnis — es zeigt, dass unser Modell den theoretisch optimalen Detektor bereits nichtparametrisch einholt und dabei robuster ist. Temperatur/top-p-korrigierte P für den NP-LR wäre der einzige Rettungsversuch, aber: Lücke groß (0.36 vs 0.4255, +0.065 nur zum Gleichstand), Temperatur-Konditionierungstest war null, weiterer GPU-Job mit unbekanntem exaktem T. Erwartungswert zu niedrig → nicht verfolgt. `USE_NPLR` bleibt per Default aus; Champion sub_006 unverändert.
