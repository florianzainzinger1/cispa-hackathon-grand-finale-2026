# Session-Chronik & Lagebericht — Task 1: Text Watermark Localization

**CISPA Hackathon Grand Finale, 2. Juli 2026 — Team !!1337**
Dieses Dokument fasst die komplette Session zusammen: Verlauf, alle Versuche, deren Bedeutung und die Begründung jeder Schlussfolgerung. (Kondensierte Chronik, kein wörtliches Transkript; Details in den Reports 01–10.)

---

## 1. Aktuelle Lage (Stand: nach Research- & Endspiel-Phase)

**Leaderboard — wir sind von #1 auf #4 gefallen:**

| Rang | Team | Public Score |
|---|---|---|
| 1 | Syntax Terror | **0.4005** |
| 2 | SeeWeedZ | 0.3811 |
| 3 | We Are All Alex | 0.3444 |
| **4** | **!!1337 (wir)** | **0.3362** |
| 5 | SarAI | 0.3256 |

**Der entscheidende Befund:** Wir waren stundenlang #1 (Peak 0.3362). Dann sind **drei Teams gleichzeitig um +0.04 bis +0.06 gesprungen** (SeeWeedZ 0.338→0.381, Syntax Terror 0.24→0.40, Alex 0.31→0.344). Ein simultaner Multi-Team-Sprung dieser Größe ist statistisch **kein unabhängiges Finden** → es kursiert **geteilte Information** (mit hoher Wahrscheinlichkeit im Discord), die einen strukturellen Hebel enthält, den wir nicht haben. Offene Aufgabe: diese Information beschaffen (Discord lesen).

- Deadline: **morgen 10:00**. Wertung: laut Team-Einschätzung **Regel (a)** — privater Score der besten Public-Submission. ⇒ Unser Task-1-Finale ≈ die private Güte unseres Modells, die feststeht (Plateau).
- Zählender Eintrag bleibt **sub_006** (alle späteren Züge lagen public darunter).

**Unsere Submission-Historie:**

| # | Submission | Kern-Idee | Val @0.1%FPR | Public |
|---|---|---|---|---|
| 73 | sub_001 random | Pipeline-/Endpoint-Test | – | 0.0009 |
| 113 | sub_002 min(L,R)-Fusion | Fenster-p-Werte, Boundary-Bleed-Fix | 0.0764 | 0.0319 |
| 130 | sub_003 HMM | 5-State-Posterior statt Fenster | 0.2351 | 0.1568 |
| 137 | sub_004 HMM getuned | Transitionen + Train+Val-Fit | 0.2900 | 0.1901 |
| 164 | sub_005 + Entropie | Emissionen pro Entropie-Level normalisiert | 0.3762 | 0.2831 |
| 175 | **sub_006 + Logprob** | Gumbel/TextSeal auf Token-Logprob konditioniert | **0.4255** | **0.3362** ← zählender Eintrag |
| 311 | sub_007 Rank-Konditionierung | Gumbel auf Rang statt Logprob | 0.4287 | 0.3269 |
| 499 | contingency eps2 | eps=2.0 Emissions-Glättung (Endspiel-Zug) | 0.4266 | 0.3357 |
| 516 | sub_008 eps2+uni151643 | eps2 + Unigram-vocab-Fix (Endspiel-Zug) | 0.4260 | 0.3349 |

Endspiel-Erkenntnis: Unsere validation-gleichwertigen Varianten ziehen public **konsistent in [0.327, 0.336]** — SeeWeedZ/Syntax Terror liegen bei 0.38–0.40. Das ist **kein Varianz-Problem**, sondern ein echter Modell-Rückstand.

---

## 2. Chronologischer Verlauf

1. **Setup (Vormittag):** JuDoor-SSH-Key-Update (IP-Range-Problem), Entdeckung der neuen TOTP-2FA auf JURECA, Lösung über WSL-SSH-Multiplexing (Windows-OpenSSH kann kein ControlMaster). Teammate-Setup im Team-Ordner von zainzinger1. Alle Task-Daten (Datensätze, `watermark_config.yaml` mit **allen Keys und Detektor-Parametern**, Templates) beschafft; die drei offiziellen Detector-Repos an den gepinnten Commits geklont (lokal + Cluster).
2. **Verständnis & Strategie:** Projekt-Overview (Report 01), Strategie-Dokument (Report 02) mit zwei externen Reviews (ChatGPT-Vergleich + Einwände-Review) abgeglichen; Kernentscheidungen: p-Werte als dokumentübergreifende Währung, min-p-Fusion, analytisch-vor-ML, Messgerät-zuerst.
3. **Phase 0:** Metrik (gegen sklearn verifiziert), Validator, Label-EDA (Spans ≥24 Tokens, ≤6/Dok, Train/Val strukturidentisch). Random-Submission bestätigte, dass Server-Metrik = unsere Metrik (0.0009 ≈ FPR-Target).
4. **Phase 1 (Signale):** Unigram (Masken-Detektivarbeit: 18 Varianten, per Binomial-Span-Tests verifiziert), KGW (CUDA-Philox by-the-book, starkes Gate: 67/291 vs. 10/269 Segmente), Gumbel/TextSeal (numpy-Port der PRF, bit-genau gegen Repo validiert — reine Integer-Arithmetik, kein CUDA nötig).
5. **Phase 2–3:** Fenster-p-Werte → Fusion. Entdeckung des Tages #1: **Boundary-Bleed** — die "False Positives" waren Clean-Tokens direkt neben echten Spans, deren Fenster in den Span hineinragten. Fix min(links,rechts): Val 0.013 → 0.076.
6. **Phase 4:** **HMM-Posterior** (5 States, faktorisierte gebinnte Emissionen, Transitionen aus EDA). Löst Boundary-Bleed UND Span-Rand-Verlust gleichzeitig: Val 0.29, Public 0.19.
7. **Phase 5:** Entdeckung #2: **Entropie-Konditionierung** (Clean-PRF ist entropie-unabhängig, WM-Signal wächst mit Entropie → Emissionen pro Level normalisieren): Val 0.376, Public 0.283 → **Platz 1**. Entdeckung #3: **Logprob-Konditionierung** für Gumbel/TextSeal (Gumbel-max-Kopplung: unwahrscheinlicher gewählter Token ⇒ extremer PRF-Wert): Val 0.4255, Public 0.3362.
8. **Gap-Extension (drei Bursts, nach SarAIs Aufholjagd):** ~15 weitere Ideen systematisch getestet — alle ≤ Rauschen (vollständige Liste unten). Qwen-Entropien/Logprobs/Ranks wurden von einem delegierten Hintergrund-Agenten auf der Cluster-GPU berechnet.
9. **Challenger-Runde (externes Review, Champion eingefroren):** Vier strukturierte Challenger mit harten Stop-Regeln (müssen Validation UND Split-Half UND Reverse-Split schlagen; keine Submission ohne explizites GO). Ergebnis: A Tail-Veto ❌ (FP-Tail ist echt mehrdeutiger Text, nicht lernbar — Subagent-Analyse), B Interval-DP ❌ (nur mit val-getuntem τ minimal besser = Protokollverstoß), C Doc-Scheme-Suppression ❌ (trifft echte seltene Spans), D-Proxy 14B/32B a priori abgelehnt (Generator IST das 7B laut YAML), **D-Rank ✅ einziger Ship-Kandidat** (alle vier lokalen Checks positiv).
10. **sub_007 (Rank) submitted nach User-GO:** Public 0.3269 — schlechter als sub_006 (0.3362). Die Downside-Absicherung griff (Best-Counts-Regel: sub_006 bleibt zählender Eintrag, nichts verloren). Lehre: Bei diesem Metrik-Typ können selbst vierfach geprüfte lokale Gewinne von +0.003 im Public-Subset-Rauschen (±0.01) untergehen.
11. **Deep-Research-Phase (2 unabhängige Läufe):** Interne Deep-Research (105 Agenten, 23 Quellen, adversarial verifiziert) + Gemini Deep Research, beide mit demselben fokussierten Prompt. Beide konvergierten stark und lieferten zwei echte theoretische Hebel: (a) **NP-optimaler Per-Token-Detektor** (`log Σ_w u^(1/P_w−1)`, nutzt volle Qwen-Verteilung) und (b) **Tr-GoF / Higher-Criticism** (Verteilungs-Form der u-Werte statt Summe). Beide voll implementiert und getestet — **keiner schlägt den Champion** (NP-LR 0.36–0.37, GoF 0.009–0.012 standalone). Grund (jetzt strukturell verstanden): Beide „optimalen" Ansätze der Literatur lösen die **Summen-Dilution** in gemischten Dokumenten — ein Problem, das unser Per-Token-HMM konstruktionsbedingt nicht hat. Unser HMM ist bereits die suffiziente Statistik. Details: Report 12.
12. **Discord-Hinweis (Maitri, Task-Entwicklerin):** Unigram-Greenlist nutzt vocab_size=151643 (Base), Special-Tokens ausgeschlossen — wir hatten 152064. Getestet: Masken zu 50% unabhängig (unser Signal war halb Rauschen), ABER Korrektur bringt nur +0.0008 (Unigram = 3.4% der Masse, strukturell zu schwach). KGW mit 151643: bit-identisch (CUDA-randperm-Prefix-Sharing), irrelevant.
13. **Endspiel nach Überholung:** Bei Regel (a) durchdacht — validation-gleichwertige Züge ändern das Finale (≈ private P) nicht, sind aber freie Public-Lotterie. 3 Züge gezogen (eps2, sub_008), alle in [0.335, 0.336], keiner schlägt SeeWeedZ. Erkenntnis: echter Modell-Rückstand, kein Varianz-Problem.
14. **Jagd auf SeeWeedZ' Hebel:** Nach dem Multi-Team-Sprung gezielt strukturelle Hypothesen getestet: (a) KGW-Rekonstruktion — korrekt (Green-Rate skaliert sauber mit Entropie 0.28→0.48). (b) Generierungs-Temperatur/top-p — Argmax-Match-Sweep über T×top-p zeigt **keinen** Sprung Richtung 1.0 (bester Gap bei T=1.0, unserer Einstellung). (c) KGW-vocab_size 151643 — bit-identisch. **Alle drei geschlossen.** Der Hebel bleibt unidentifiziert → Discord ist die verbleibende Informationsquelle.

---

## 3. Was funktioniert hat — und warum

| Baustein | Effekt | Warum es funktioniert |
|---|---|---|
| Exakte Signal-Rekonstruktion mit den YAML-Keys | Fundament | Wir testen nicht "sieht aus wie Watermark", sondern rechnen das Watermark nach. Sanity-Gates pro Scheme fingen zwei Fast-Fehler (Unigram-Maske, KGW-CPU-Falle). |
| min(L,R)-Fensterlogik | 0.013→0.076 val | Ein Token zählt nur, wenn beide Seiten watermarked aussehen → eliminiert Boundary-Bleed, der das gesamte 0.1%-FPR-Budget fraß. |
| HMM-Posterior | 0.076→0.29 val | Per-Token-Emissionen = kein Bleed konstruktionsbedingt; Forward-Backward holt die Span-Ränder zurück, die Fenster opfern. Span-Struktur (zusammenhängend, ≥24 Tokens) steckt in den Transitionen. |
| Entropie-Konditionierung | 0.29→0.376 val | Physik: Clean-PRF-Werte sind uniform (entropie-unabhängig), WM-Signalstärke wächst mit Entropie (gemessen: Gumbel 1.12/1.22/1.48). Normalisierung *pro Level* verhindert, dass die Text-Entropie selbst leakt. |
| Logprob-Konditionierung (Gumbel/TextSeal) | 0.376→0.4255 val | Gumbel-max wählt argmax u^(1/p): Ein gewählter Token mit niedriger Modell-Wahrscheinlichkeit erzwingt extremes u (gemessen: WM-Score 1.74 bei lp<−3 vs. 1.10 bei lp>−0.3; Clean flach 1.01). |

---

## 4. Was nicht funktioniert hat — Ergebnis, Bedeutung, Schlussfolgerung

| Idee | Ergebnis (val @0.1%) | Warum es scheiterte | Schlussfolgerung |
|---|---|---|---|
| Empirische Tail-Kalibrierung | Train 0.022 → Val **0.000** | Mit 47k Clean-Tokens kann man p<2e-5 nicht auflösen | Analytische p-Werte transferieren, empirische Tails nie |
| Roher Logprob-Emissionskanal | **0.005** (von 0.376) | Stil-Detektor, transferiert nicht zwischen Dokumenten | Modell-Features nur normalisiert-pro-Level verwenden |
| Erlang-Dauermodell (k=2,3) | 0.215–0.268 (von 0.29) | Geometrische Flexibilität hilft dem 0.1%-Operating-Point | Transitionen einfach halten |
| Feinere Bins (Score 18 / Entropie 5–6) | 0.277–0.351 | Schätzvarianz schlägt Auflösung (~10k WM-Tokens/Scheme) | Bin-Anzahl ist durch Datenmenge gedeckelt |
| Scheme-gewichtete Priors, Transition-Ensembles, p_start-Fix, eps-Tuning, max-u-Spike-Bin | alle ±0.002 | Emissionen dominieren den Posterior vollständig | Prior-/Glättungs-Achse ist ausgereizt |
| LightGBM-Stacker (leak-frei, GroupKFold) | OOF 0.276 vs. HMM 0.321 | Gelerntes Modell findet nichts über die analytischen Likelihoods hinaus; 180 Dokumente echte Diversität | „Wir haben die Schlüssel" — analytisch schlägt ML, jetzt bewiesen |
| Argmax-Match (WM-Token exakt vorhersagen) | 0.4266 (= Rauschen) | Information ist redundant zur u\|Logprob-Konditionierung | Spektakulär ≠ zusätzlich informativ |
| KGW: Logprob-/Joint-Konditionierung | 0.400–0.410 (von 0.4255) | 2-Bin-Signal × Levels = dünne, verrauschte LRs im Tail; exakte p_green-Rechnung an Self-Salt-Seeding mathematisch unmöglich | KGW ist bei δ=1.5 strukturell gedeckelt (Recall ~0.15) |
| Unigram p_green (exakte Null pro Position) | 0.4264 (= Rauschen) | Theorie perfekt bestätigt, aber strength=1.0 ⇒ LR/Token 1.2–1.6 akkumuliert nie über die 0.1%-Schwelle | Unigram (~3.4% der Masse) ist für diese Metrik praktisch unerreichbar |
| TextSeal Dual-Key-2D-Emission | 0.4125 | 16 Joint-Bins × Levels zu dünn; Fusion+Konditionierung trägt die Info bereits | Geschlossen |
| Viterbi-Blend (Grenzen schärfen) | 0.4238–0.4244 | Posterior ist bereits die bessere Ranking-Statistik | Geschlossen |
| Semi-supervised EM auf Test (1.39M Tokens), soft + hard (conf 0.9/0.95/0.99) | 0.363–0.418 | Sichere Pseudo-Labels = nur Span-Kerne → selektionsverzerrte WM-Verteilungen | Selbst-Training degradiert bei dieser Problemstruktur |
| Tail-Veto-Classifier (nur kritische Score-Zone, Subagent) | +0.0008 max, Split-Half negativ | Nur ~250 Clean-Zone-Tokens; max. Veto-Konfidenz 0.26; Features tragen dieselbe Info wie die Emissionen | Der FP-Tail ist echt mehrdeutiger Text, kein lernbarer Junk |
| Constrained-Interval-DP (≤6 Spans, ≥24 Länge, ≥12 Gap) | +0.002 nur bei val-getuntem τ | Harte Constraints fügen dem weichen Posterior nichts hinzu | Geschlossen |
| Per-Dokument-Scheme-Suppression | 0.419 (von 0.4255) | Unterdrückt echte seltene Spans stärker als FPs | Geschlossen |
| Rank-Konditionierung (sub_007) | val +0.003, Splits +0.02–0.03, **public −0.009** | Lokal robust, aber Public-Subset bevorzugt Logprob; Delta im Rauschband | Lokale Mini-Gewinne (<0.005) sind auf 30%-Subsets nicht nachweisbar; sub_006 bleibt Champion |
| **NP-optimaler LR-Detektor** (Deep-Research #1, volle Qwen-Verteilung) | val 0.36–0.37 (von 0.4255) | Per-Token-Optimalität ≠ span-gepoolte Metrik; unsere Logprob-Konditionierung approximiert ihn bereits nichtparametrisch & robuster | Theoretisch optimal ≠ besser bei dieser Metrik/Architektur |
| **Tr-GoF / Higher-Criticism** (Deep-Research #1 von Gemini) | standalone 0.009–0.012 | GoF schlägt SUMMEN-Detektoren; unser HMM nutzt keine Summe, sondern Per-Token-Likelihood | Redundant zum HMM (dreifach belegt mit NP-LR) |
| Unigram vocab_size 151643 (Discord-Fix) | val +0.0008 | Unigram = 3.4% der Masse, strukturell zu schwach (strength=1.0) | Echter Fix, aber competitively irrelevant |
| α=0.1 Dual-Key-Fusion (Deep-Research) | val 0.31–0.40 (von 0.4255) | 50/50-Fusion ist bereits optimal | Geschlossen |
| Generierungs-Temperatur/top-p (Argmax-Match-Sweep) | Match-Rate flach ~0.64 über alle T×top-p | Unser p bei T=1 ist bereits optimal fürs Argmax; kein besseres Generierungs-p findbar | Temperatur endgültig ausgeschlossen (2. unabhängiger Test) |
| Endspiel-Lotterie-Züge (eps2, sub_008) | public [0.335, 0.336] | Echter Modell-Rückstand zu Top-Teams, kein Varianz-Problem | Züge zwecklos gegen genuines besseres Modell |

---

## 5. Die Schlussfolgerungs-Kette („warum Plateau")

1. Die drei großen Gewinne kamen jeweils aus einer **neuen Informationsquelle** (Span-Struktur → HMM; Modell-Entropie; Token-Logprob), nicht aus Tuning.
2. Alle verbleibenden Informationsquellen wurden geprüft: KGW-Verbesserung (mathematisch unmöglich bzw. empirisch schädlich), Unigram (Scheme zu schwach), Argmax-Determinismus (redundant), unbeschriftete Testdaten (Selektionsbias), gelernte Fusion (verliert), 2D-Emissionen (zu dünn).
3. Jede Tuning-Achse (Bins, Levels, Transitionen, Priors, Glättung, eps) zeigt ein flaches Plateau um 0.4255 ± 0.002.
4. **⇒ ~20 konsekutive, gut motivierte Negativergebnisse (inkl. der vier strukturierten Challenger des externen Reviews) sind der empirische Beweis, dass sub_006 das Optimum dieser Lösungsklasse ist.** Weitere Task-1-Stunden haben Erwartungswert ≈ 0.
5. **Zusatzlektion aus sub_007:** Auch ein Kandidat, der vier unabhängige lokale Robustheits-Checks gewinnt, kann auf dem 30%-Public-Split verlieren, wenn sein Validation-Vorsprung (+0.003) kleiner ist als das Subset-Rauschen (±0.01). Konsequenz: Verbesserungen unterhalb von ~+0.01 Validation sind bei dieser Metrik nicht zuverlässig nachweisbar — die Messlatte für weitere Submissions liegt entsprechend hoch. Dank Best-Counts-Regel war der Versuch risikofrei (sub_006 blieb zählender Eintrag).

**Submission-Politik (Varianz-Argument):** Es zählt Best-Public, aber das Finale entscheidet der Private-Score dieser Best-Public-Submission. Eine nur-gleichgute Variante, die zufällig public gewinnt, ersetzt unseren soliden Finaleintrag durch einen Lotterieschein. Darum gehen nur echte Validation-Verbesserungen raus. Unser Modell ist bewusst varianzarm (analytische p-Werte, breite Plateaus, kein empirisches Tail-Fitting) — der Public-Vorsprung sollte privat halten.

**Bedrohungsanalyse:** SarAI klettert unsere Kurve nach, wird aber dieselben Plateaus treffen. Die letzten +0.05 zwischen 0.28 und 0.34 stecken in den zwei Konditionierungs-Insights — die findet man nicht durch Standard-Vorgehen.

---

## 5b. Update — der Rückfall auf #4 und was er bedeutet

Wir haben von 0.0009 auf 0.3362 gearbeitet und waren lange #1. Der Rückfall auf #4 kam NICHT durch einen Fehler unsererseits, sondern durch einen **simultanen Multi-Team-Sprung** (+0.04 bis +0.06), der auf geteilte Information hindeutet. Unser Modell ist über ~30 Experimente (eigene + zwei Deep-Research-Läufe + Discord-Hinweis) nachweislich am Plateau seiner Lösungsklasse. SeeWeedZ/Syntax Terror haben einen **strukturellen Hebel außerhalb dieser Klasse** — den wir trotz gezielter Suche (KGW-Rekonstruktion, Generierungs-Temperatur, vocab_size) nicht identifizieren konnten. Die verbleibende Informationsquelle ist der **Discord** (wo der Multi-Team-Sprung koordiniert wurde).

**Was wir richtig gemacht haben und was bleibt:** Die gesamte Pipeline ist sauber, dokumentiert und reproduzierbar; jede Entscheidung ist mit Zahlen belegt; alle Sackgassen sind festgehalten (spart jedem, der übernimmt, Stunden). Unser Modell ist konservativ/robust — auf dem Private Split könnten wir besser abschneiden als der Public-Rang 4 vermuten lässt.

## 6. Empfehlungen (Stand jetzt)

1. **SOFORT: Discord lesen.** Der Multi-Team-Sprung (+0.04–0.06 bei 3 Teams) = geteilter struktureller Hebel. Das ist der einzige noch nicht ausgeschöpfte Informationskanal. Blindes Testen hat alle konkreten Hypothesen erschöpft.
2. **Task 2/3** — der Hackathon wird über alle Tasks gewertet; bei Task 1 kämpfen wir bergauf gegen ein besseres Modell, bei Task 2/3 liegen Punkte offen.
3. **Jury-Präsentation** — die Story ist vollständig dokumentiert und ungewöhnlich beweisbar (jede Entscheidung mit Zahlen, ~30 Experimente inkl. zweier Deep-Research-Läufe).
4. **Falls der Discord-Hebel gefunden wird:** Pipeline ist bereit, Umsetzung in Minuten möglich (Cache + HMM stehen).

*Alle Details: Reports 01–10 in diesem Ordner, Submission-Log in `../submissions/submissions_log.md`.*
