# Report 15 — Full Audit: Wie Task 1 bearbeitet wurde, was underperformt, und warum

**Jul 3, ~00:00.** Vollständige Untersuchung von Modell, Prozess und Statistik.
Frische Diagnostik: `src/diag_full_audit.py` (Miss-Attribution, Nats-Budget,
FP-Struktur, Oracle-Bounds, Doc-Bootstrap). Basis: Champion (fit train, cond best,
trans 0.03/80) auf Validation — val 0.4384, public 0.3744.
Rahmendaten: 44.325 WM- / 51.379 Clean-Tokens, 396 Spans, **FP-Budget = 51 Tokens**.

---

## A. Modell: Wo es underperformt — und wie

### A1. Scheme-Asymmetrie ist Informations-Physik, kein Implementierungsfehler

Miss-Attribution pro WM-Token (Champion, val):

| Scheme | WM-Tokens | missed | davon dead-span | edge≤3 | low-ent | invalid | other (in-span) |
|---|---|---|---|---|---|---|---|
| unigram | 3.286 | **97.3%** | 89.1% | 0.9% | 1.2% | 0.4% | 5.8% |
| kgw | 11.605 | **83.1%** | 38.4% | 2.7% | 12.3% | 1.1% | **28.7%** |
| gumbel | 14.703 | 41.2% | 17.1% | 5.0% | 7.6% | 2.3% | 9.1% |
| textseal | 14.731 | 40.6% | 13.0% | 5.2% | 8.0% | 0.7% | 13.8% |

Dead spans (Span-Maximum < τ): unigram **31/35**, kgw **48/92**, gumbel 30/135, textseal 30/134.

Das Nats-Budget (E[LLR|wm] aus den gefitteten Emissionen × Median-Spanlänge) erklärt es
abschließend — ein Span braucht grob ln(1000)≈6.9 + Entry-Kosten ≈4.9 ≈ **12 nats** zum Feuern:

| Scheme | nats/Token | × Median-Span | Budget |
|---|---|---|---|
| unigram | 0.025 | ×63 | **1.6** — hoffnungslos (Faktor 7 zu wenig) |
| kgw | 0.058 | ×95 | **5.5** — knapp unter Schwelle: nur überdurchschnittliche (High-Entropy-)Spans feuern |
| textseal | 0.212 | ×95 | 20.1 — komfortabel |
| gumbel | 0.337 | ×95 | 32.0 — komfortabel |

**Konsequenz:** 37% der WM-Masse (kgw+unigram) ist mit *jedem* Per-Token-Likelihood-Ansatz
strukturell unter- oder nicht erreichbar. Der kgwpg-Test (Report 14) hat bewiesen, dass
selbst die exakte Null das nicht ändert (KGW-Recall 0.19→0.28, aber nicht transferabel und
von den starken Kanälen über den Threshold zurückbezahlt).

### A2. Der eigentliche Engpass: Boundary-Bleed frisst 94% des FP-Budgets

Das gesamte 0.1%-Budget sind **51 Clean-Tokens**. Davon: **94.1% innerhalb 10 Tokens eines
echten Spans, 54.9% innerhalb 3, 100% in WM-haltigen Dokumenten.** Oracle-Treppe (Clean-Scores
in Boundary-Zone genullt → Threshold entspannt sich):

| Oracle-Zone | TPR@0.1% | Headroom |
|---|---|---|
| Champion | 0.4384 | — |
| ≤3 Tokens (Posterior-Ausläufer-Kern) | 0.4544 | +0.016 |
| ≤5 | 0.4736 | +0.035 |
| ≤10 | 0.5065 | **+0.068** |
| ≤20 | 0.5172 | +0.079 |

**Das ist das größte quantifizierte Headroom im ganzen Problem — in der Größenordnung des
Gaps zu SeeWeedZ (+0.094 public).** Mechanismus: Der Forward-Backward-Posterior läuft nach
Span-Ende langsam aus (Exit-Rate 1/80 pro Schritt, evidenzarme Folge-Tokens bremsen den
Abfall). Unsere Gegenversuche — Tail-Veto (max. Konfidenz 0.26), Interval-DP, Viterbi-Blend,
Doc-Suppression — versuchten alle, den Tail zu *lernen* oder *hart* zu schneiden, und
scheiterten. Nicht erschöpft: strukturelle Boundary-Modellierung (asymmetrische/
evidenzgekoppelte Exit-Transitionen, Randzonen-States). Achtung: jede Schärfung des Randes
bezahlt mit Edge-Recall (aktuell ~5% Edge-Misses bei den starken Schemes) — der Trade-off
wurde nie systematisch vermessen. **Rückblickend haben wir Stunden in die 37%-KGW/Unigram-
Sackgasse investiert, während das Boundary-Headroom größer und unbeackert war.**

> **ADDENDUM (Jul 3, ~01:00, unabhängiger Fresh-Context-Researcher): Die Boundary-These
> ist GEMESSEN TOT — das +0.068-Oracle ist ein Label-Oracle-Artefakt.** Vollständige
> Transform-Batterie gegen den Champion (Two-Filter-min(fwd,bwd), Min-Duration-HSMM
> D∈{12,24}, evidenzgekoppelte Entry/Exit-Transitionen, Erosion, Shoulder-Clip, Sag-Fill):
> alles ≤ Champion außer Sag-Fill k=5 (+0.0025 — Faktor 10 unter der Beweisbarkeits-
> schwelle). Mechanistische Begründung, dreifach: (1) **Labels sind scharf** —
> Superposed-Epoch über alle 396 Spans: Null WM-Signal außerhalb der Labels (llr springt
> exakt bei r=0 von −0.15 auf +0.13). (2) **Die 51 Budget-FPs sind selektions-gepickte
> Zufalls-Evidenz**: ihre own-scheme-llr (med +0.31) ist von echten WM-Tokens (med +0.22)
> ununterscheidbar — jede observablen-basierte Demotion-Regel wird per Konstruktion
> getäuscht (erklärt in einem Zug, warum Tail-Veto/Interval-DP/Viterbi/Suppression/alle
> 7 neuen Transforms ~1:1 statt 1:63 tauschen). (3) **Refill-Zensus**: τ auf das
> ≤10-Oracle-Niveau zu senken erfordert ~140 demote-te Clean-Tokens, davon ~50 weiter
> als 20 Tokens von jedem Span — Zufalls-Runs, die das Budget sofort wieder auffüllen.
> Literatur (WaterSeeker, WISER, Adaptive Testing 2025, CPD-Segmentierung) = alles
> Scan-Statistik-Klasse, bereits tot per Report 14. ⇒ F2 ist hinfällig; Empfehlung F1
> (Task 3) gilt uneingeschränkt.

> **ADDENDUM 2 (Jul 3, ~01:30, `src/boundary_refine.py`): WaterSeeker-Track explizit
> implementiert und unabhängig bestätigt tot.** HMM-Proposals (Posterior>0.5) + lokale
> Intervall-Traversierung (sum/√len, R∈{10..30}, Lmin∈{10,16,24}) + weiche Randabwertung
> (α∈{0..0.5}) sowie Proposal-Edge-Erosion (k∈{2,3,5,8}×α): **kein einziger Budget-FP
> verschwindet in irgendeiner Konfiguration** (FP≤3/5/10/20 = 29/39/49/51 von 52,
> konstant). Das Refinement BEHÄLT die Bleed-Tokens im Kern — ihre Zufallsevidenz
> maximiert die Intervall-Statistik mit; Erosion trifft ausschließlich echte Edge-Tokens
> (Edge-Recall≤3 0.07→0.04; val −0.009 bei k=8); Surgical-Rescoring −0.07. Best case
> Erosion k=2: −0.0002. Zweiter unabhängiger Beweis von Punkt (2): die FPs sind
> beobachtungs-ununterscheidbar. Boundary-Achse doppelt geschlossen.

### A3. FP-Konzentration: die Metrik hängt an Einzeldokumenten

Die 51 Budget-Tokens verteilen sich auf **nur 8 Dokumente; die Top-3 halten 69%**
(validation_24 allein 15 = 29%). Konsequenzen: (a) Half-Splits streuen extrem
(halfA 0.53 vs. halfB 0.38 beim Champion — je nachdem, wo die Pathologie-Docs landen),
(b) jede Modelländerung, die EIN pathologisches Doc anders behandelt, verschiebt die
Gesamtmetrik um Prozentpunkte.

### A4. In-Span-Lücken

„other"-Misses (Span feuert, Token mittendrin nicht): **28.7% bei KGW**, 9–14% bei
gumbel/ts. Der Posterior sackt mitten in live Spans ab, wo lokale Evidenz-Durststrecken
auf die geometrische Dauerannahme (mean 80) treffen. Erlang-Dauern wurden getestet
(schlechter am Operating-Point) — evidenzadaptive Glättung innerhalb erkannter Spans nicht.

### A5. Metrik-Varianz: fast nichts war je entscheidbar

Doc-Bootstrap (1000 Iterationen): Champion **CI95 [0.374, 0.500], sd 0.033**. Paired-Delta
kgwpg-E7: Mittel −0.005, CI [−0.150, +0.034], P(Δ>0)=0.61 — der „+0.0069-Gewinn" war nie
signifikant, und sein linker Tail (−0.15!) zeigt das Kollaps-Risiko in ungünstigen
Doc-Kompositionen. Faustregel ab jetzt: **val-Unterschiede < +0.03 sind mit 90 Docs
prinzipiell nicht auflösbar** (public mit ~396 Docs: sd ≈ 0.016). Das erklärt val→public-
Faktor-Streuung (0.66–0.79) und warum lokal „bewiesene" Mini-Gewinne (rank +0.003) public
verloren: sie waren nie Messungen, sondern Rauschen.

---

## B. Prozess: Wie gearbeitet wurde — und die Fehler

**Zeitachse (aus Report-/Submission-Timestamps):** Setup/Datenbeschaffung bis 14:12 ·
Metrik+EDA 14:12–15:00 · exakte Signale 15:00–15:30 · HMM+Konditionierungen 15:30–16:10
(**sub_006 = 0.3362, Platz 1, nach ~2h Modellierung**) · 16:10–20:16 Gap-Extension+Research
(+0.038 via v2) · 20:16–00:00 strukturelle Jagd (Ertrag 0). Klassisch abnehmender
Grenzertrag: 80% des Scores in den ersten 2 Modellierungs-Stunden.

1. **Der teuerste Einzelfehler: unvollständige Repo-Lektüre.** `scoring_method="v2"` ist
   der *Default* in `textseal/watermarking/detector.py` (Z. 358) — derselben Datei, aus der
   wir die PRF portiert haben. Gefunden wurde v2 erst ~20:00 über einen Gemini-Research-Lauf,
   nachdem drei Teams gesprungen waren. Vollständige Lektüre der Detector-API in der
   Signal-Phase (~15:00) hätte +0.036 fünf Stunden früher und vermutlich Platz 1 gehalten.
   `localized_detect` stand in derselben Datei (für uns wertlos, aber die Konkurrenz hat
   diese Datei offensichtlich gelesen).
2. **Cache-Staleness-Trap:** Die Parquet-Caches trugen nach dem v2-Sprung noch v1-Flags;
   `v2_sweep` erklärte reale Gewinne stundenlang zu „nicht reproduzierbar" (falsche
   Plateau-These), bis `dedup_audit.py` die Trajektorie exakt reproduzierte. Lehre:
   Cache-Provenance versionieren, nicht vermuten.
3. **Leaky-Val als Entscheidungsgrundlage:** Submissions fitten train+val, getuned wurde
   auf val. Die berichteten sub_010-val-Zahlen (0.4316/0.4384/0.4431) waren leaky;
   sub_010_rank wurde auf dieser Basis geshippt und verlor −0.016. Rank wurde damit
   **zweimal** geshippt (sub_007, sub_010_rank), obwohl das erste Debakel dokumentiert war.
   Das Ship-Gate (Split-Half + Reverse) entstand erst *danach*.
4. **Schätzungen wurden zu Fakten:** „KGW exact LR ist +0.02–0.04 wert" stand als
   Quasi-Fakt im Memory — gemessen: +0.007, nicht robust, bootstrap-insignifikant.
   Ceiling-Schätzungen und Messungen wurden nicht konsequent getrennt ausgewiesen.
5. **Bookkeeping-Lücke:** Public-Best 0.3744 stammt von einer im Log nicht erfassten
   Submission (sub_010_lp traf auf „previous_best 0.3744"). Für die Wertung egal
   (Best-Counts), für die Rekonstruktion ärgerlich.
6. **Parallel-Session-Races:** Eine Test-Extraktion (8 Shards, GPU) wurde gestartet,
   *bevor* das Gate-Verdict vorlag, und halbfertig gekillt (tote Shard-Dateien auf dem
   Cluster). Memory-Writes kollidierten mehrfach. Es gab keinen Koordinationsmechanismus
   zwischen Sessions außer dem Memory-File.
7. **Der Discord-Kanal — der größte strategische Fehler:** Der Multi-Team-Sprung (+0.04–0.06
   simultan bei 3 Teams) war ein *Informations*-Ereignis, kein Modell-Ereignis. Niemand hat
   den Kanal kontinuierlich gelesen; ein einziger dort abgeholter Hinweis (Maitri/vocab_size)
   wurde spät und passiv aufgenommen. Stunden gingen in Blindhypothesen-Tests
   („Jagd auf SeeWeedZ' Hebel"), die ein Discord-Read in Minuten hätte klären können.
8. **Kleinere Punkte:** hmm.py-Kommentar „sampled at temp 0.9" widerspricht der Chronik
   („T=1.0 optimal, Temperatur ausgeschlossen") — nie aufgelöst. Split-Halves ad hoc
   (alternierend statt hash-basiert). Val→public-Faktor nie modelliert. 0-Byte-Upload-Falle
   kostete einen Extraktions-Anlauf.

---

## C. Statistisches Grundproblem (Meta-Lehre)

90 val-Docs, 51-Token-FP-Budget, FP-Konzentration in 8 Docs ⇒ die Validierungs-Metrik hat
sd ≈ 0.033. **Ein Großteil der ~35 „Negativergebnisse" ist strenggenommen „unentscheidbar
klein" statt „widerlegt"** — für Rang-Entscheidungen äquivalent (nichts davon hätte
zuverlässig +0.03 gebracht), aber die Berichtssprache („geschlossen") suggeriert mehr
Sicherheit, als die Statistik hergibt. Das Ship-Gate war die richtige Antwort, kam aber
erst nach zwei vermeidbaren Fehlschüssen.

---

## D. Kompetitive Einordnung

- **Wo SeeWeedZ plausibel gewinnt:** Boundary-Handling ist der einzige identifizierte
  Hebel, dessen Headroom (+0.07–0.08 val) die Gap-Größenordnung erreicht — nicht KGW
  (durch Messung geschwächt) und nicht Scan/native-Detection (gemessen tot).
  Alternativ: ein unbekannter Discord-Hebel.
- **Intel:** Syntax Terror wurde beim Extrahieren desselben keyed-LR-Features beobachtet
  (Login-Node ps, `003-mk-keyed-lr`) — sie shippen möglicherweise in genau das Artefakt,
  das unser Gate abgefangen hat.
- **Private-Risiko unseres Eintrags:** sub_010_lp ist varianzarm designt (analytische
  p-Werte, keine empirischen Tails) — aber die FP-Konzentration (8 Docs) bedeutet, dass
  auch der Private-Score (70%, ~920 Docs) eine Doc-Kompositions-Lotterie mit sd ~0.01
  bleibt. Erwartung: Private ≈ Public ± 0.02.

## E. Was gut lief (der Vollständigkeit halber)

Exakte Signal-Rekonstruktion mit Sanity-Gates (fing zwei Fast-Fehler) · HMM-Architektur
(löste Boundary-Bleed *des Fensteransatzes* und Span-Ränder simultan) · die zwei
Konditionierungs-Insights (Entropie, Logprob = +0.14 val zusammen) · analytisch-vor-ML
konsequent durchgehalten und mit dem Stacker-Test belegt · ab sub_007-Lektion: Ship-Gate,
Best-Counts-Disziplin, Contingency-Vorbereitung · lückenlose Doku (15 Reports, jede
Entscheidung mit Zahlen — dieses Audit war nur deshalb in einer Stunde möglich).

## F. Konsequenzen

1. **Für die verbleibenden Stunden:** Task 3. Task 1 nur bei externem Hinweis anfassen.
2. **Falls doch Task 1:** einziger offener Hebel mit ausreichendem Headroom =
   Boundary-Zone-Modellierung (A2) — aber unter dem A5-Vorbehalt, dass Gewinne < +0.03
   lokal nicht beweisbar sind; realistisch nur mit sehr großem Effekt sinnvoll.
3. **Für künftige Wettbewerbe:** (a) Detector-/Repo-APIs *vollständig* lesen, bevor man
   portiert; (b) Community-Kanäle als kontinuierliche Informationsquelle staffen;
   (c) Metrik-Varianz am Tag 1 quantifizieren (Bootstrap) und die Beweislatte danach
   ausrichten; (d) Cache-Provenance versionieren; (e) Schätzung ≠ Messung kennzeichnen;
   (f) Session-Koordination über ein Lock-/Status-File.
