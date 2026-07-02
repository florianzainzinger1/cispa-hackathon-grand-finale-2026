# Submitting Task 3

The built, validated submission lives on JURECA at:

```
/p/scratch/training2625/zainzinger1/1337/recon_work/outputs/submission.pt
```

- dict with keys `model1`..`model12`, each `Tensor(128, 3, 64, 64)` float32 in `[0,1]`
- 128 unique images per model, all finite, file ~75 MB (limit 300 MB)
- re-validate any time: `python validate_submission.py <path>`

## How to submit (you run this — it needs your API key)

The organizer endpoint is `POST http://<SUBMIT_SERVER>/submit/21-fl-audit` with an
`X-API-Key` header. Use the organizer's `task_template.py` (set `SUBMIT = True`,
`API_KEY = <your key>`, `FILE_PATH = .../submission.pt`) or:

```bash
curl -s -X POST http://<SUBMIT_SERVER>/submit/21-fl-audit \
  -H "X-API-Key: <YOUR_API_KEY>" \
  -F "file=@/p/scratch/training2625/zainzinger1/1337/recon_work/outputs/submission.pt"
```

I do not store or handle the API key — please run the submission yourself.

## Per-model reconstruction method

| model | family | method | expected quality |
|---|---|---|---|
| 1,4,5 | MLP | first-layer analytic (gradW/gradB → image) | high |
| 8 | MLP | analytic; low leakage (small gradient) | medium/noisy |
| 6,10,12 | CNN | fc1 analytic → conv inversion → RGB | high |
| 2,7 | CNN | smoothed low-res conv inversion | low (blurry) |
| 3 | CNN | conv inversion, 8×8 feature | low (blurry) |
| 9,11 | ViT | face-bank fallback (gradient matching gives patch-noise) | weak/generic |

A timestamped backup of the first valid version is at
`recon_work/outputs/submission_v1_bank.pt`.
