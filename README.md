# student-health-kaggle-sandbox

A [`kitchen`](https://github.com/rkoren/kitchen-platform) project for Kaggle
**[Playground Series S6E7 — Predicting Student Health Risk](https://www.kaggle.com/competitions/playground-series-s6e7)**.

A sandbox competition used to dogfood the platform end-to-end: scaffold → ingest → train →
promote → submit, driven by a single `menu.yaml`.

## The problem

- **Task:** multiclass classification of `health_condition` into `at-risk` / `unhealthy` / `fit`.
- **Metric:** **balanced accuracy** (mean per-class recall) — chosen because the classes are
  heavily imbalanced (`at-risk` ≈ 86%, `unhealthy` ≈ 8%, `fit` ≈ 6%). Predicting the majority
  class everywhere scores only 0.333.
- **Data:** ~690k train / ~296k test rows; 7 numeric health metrics + 7 low-cardinality
  categorical columns, all with missing values.
- **Submission:** `id,health_condition` with the string class label per row.

## The baseline

An XGBoost `multi:softprob` classifier tuned for the metric and the imbalance:

- **Class balancing** — `sample_weight = compute_sample_weight("balanced", y)` so the majority
  class doesn't swamp training. This is the load-bearing choice for balanced accuracy.
- **Categoricals** — the 7 categorical columns become pandas `category` dtype with fixed,
  explicit category lists (identical on train and test) and are fed to XGBoost via
  `enable_categorical=True`. Numeric columns pass through untouched (XGBoost handles NaN).
- **Target encoding** — `health_condition` is label-encoded to 0/1/2 (XGBoost rejects string
  labels) and decoded back to strings at submission time.

Validation (stratified 20% hold-out): **balanced accuracy ≈ 0.950** (accuracy 0.938, macro-F1
0.865, macro ROC-AUC 0.983). The primary metric `val_balanced_accuracy` drives the leaderboard
and `--auto-promote`.

## Layout

| Path | What |
|---|---|
| `menu.yaml` | Single project manifest — data source, columns, model knobs, primary metric. |
| `src/features/run.py` | Feature engineering (`engineer()` shared by train and submission). |
| `src/train/run.py` | XGBoost trainer — balanced weights, logs `val_balanced_accuracy`. |
| `src/evaluate/run.py` | Scores the champion on the hold-out split. |
| `flows/generate_submission.py` | Champion → `submissions/submission.csv` (decodes labels). |

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # installs rkoren-kitchen from PyPI

kitchen ingest                   # download the competition data → data/raw/ (needs Kaggle creds)
kitchen menu run                 # features → train (auto-promote) → evaluate
kitchen leaderboard              # rank runs by val_balanced_accuracy

python -m flows.generate_submission   # write submissions/submission.csv from the champion
kitchen submit --dry-run              # validate the CSV (no upload)
kitchen submit --wait                 # upload to Kaggle and poll for the leaderboard score
```

## CI

`.github/workflows/train-evaluate.yml` runs train → evaluate on every push/PR. It needs
`KAGGLE_USERNAME` and `KAGGLE_KEY` as repo secrets (Settings → Secrets → Actions) to ingest the
data; without them the ingest step fails. Set `ci.auto_submit: true` in `menu.yaml` (or use the
workflow's manual `submit` toggle) to submit to Kaggle from CI.
