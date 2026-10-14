# Task 1 — Text Watermark Localization (Team !!1337)

**Stand Jul 2, ~23:30. Public best 0.3744 (#3). Champion-Submission: `submissions/sub_010_v2trans_lp.jsonl`.**

## Schnellstart

```bash
pip install -r data/requirements.txt   # numpy, pandas, pyarrow, requests
# Champion reproduzieren (fit train+val, v2-Caches, trans 0.03/80):
python src/predict_hmm.py --fit train validation --split test \
    --cond best --p-enter 0.03 --mean-span 80 --out submissions/my_sub.jsonl
python src/validate_submission.py submissions/my_sub.jsonl
python src/submit.py --file submissions/my_sub.jsonl   # braucht repo-root .env mit API_KEY
```

## Wo steht was

| Datei | Inhalt |
|---|---|
| `reports/11_session_chronik.md` | **Hier anfangen** — komplette Story, alle ~30 Experimente, warum-Plateau-Kette |
| `reports/13_structural_hunt.md` | Per-Scheme-Diagnose: gumbel/ts gut (0.6), KGW/unigram threshold-gedeckelt |
| `reports/14_kgwpg_and_scan.md` | Letzte zwei Hebel gemessen & geschlossen (KGW exact LR = Artefakt; TextSeal `localized_detect` = tot) |
| `submissions/submissions_log.md` | Jede Submission mit Hypothese, val, public |
| `src/hmm.py` | Das Modell: 5-State-HMM, faktorisierte konditionierte Emissionen |
| `src/eval_kgwpg3.py` | **Ship-Gate-Template**: Kandidat muss val UND beide Split-Halves UND beide Reverse-Splits schlagen |
| `cache/*_signals.parquet` | Per-Token-Signale aller 4 Schemes (v2-dedup!) + Qwen entropy/logprob/rank |

## Die 3 Dinge, die man wissen muss

1. **Champion = v2-dedup + Konditionierung**: uni/kgw auf Entropie, gumbel/textseal auf Token-Logprob, Transitionen (0.03, 80). Val 0.4384 (fit train), public 0.3744.
2. **Gains < +0.01 val sind auf dem 30%-Public-Split nicht nachweisbar** (±0.01 Rauschen). Zweimal bewiesen (sub_007 rank, sub_010_rank). Deshalb: Ship-Gate (`eval_kgwpg3.py`) vor jeder Submission.
3. **Alle bekannten Hebel sind mit Zahlen geschlossen** (Reports 11–14): NP-LR, Higher-Criticism, Scan-Statistik, KGW exact LR, Dedup-Varianten, rank, eps, Stacker, EM, … Der 0.094-Gap zu SeeWeedZ ist mit unserer Informationsbasis nicht erklärbar — einziger offener Kanal: **Discord**.

## Cluster (JURECA)

Team-Ordner: `/p/scratch/training2625/zainzinger1/1337/watermark_localization/` —
Extraktions-Skripte + Logs in `output/` (Qwen-Signale, topk, kgw_pgreen). GPU-Extraktionen
laufen auf den Login-Node-GPUs; `output/extract_*.py` sind selbstdokumentierend.
