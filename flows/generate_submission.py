"""Generate a Kaggle submission CSV from the champion model (Playground S6E7).

Loads the champion XGBoost classifier, applies the same feature engineering to test.csv,
predicts integer class codes, and decodes them back to the string labels the competition
expects (at-risk / fit / unhealthy) → submissions/submission.csv.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

import mlflow
import pandas as pd

from kitchen.registry import get_production_uri
from kitchen.store import DataStore
from kitchen.tracking import configure_from_env

from src.features.run import CLASSES, ID, TARGET, engineer

MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "student-health-model")


def generate(params_file: str = "menu.yaml") -> None:
    with open(params_file) as f:
        params = yaml.safe_load(f)

    configure_from_env()
    store = DataStore()

    test_raw = store.load_csv(params["features"]["test_file"])
    test_df = engineer(test_raw)

    uri = get_production_uri(MODEL_NAME)
    if uri is None:
        raise RuntimeError(
            f"No champion model found for {MODEL_NAME!r}. "
            "Train with `kitchen menu run` (--auto-promote) first."
        )

    model = mlflow.xgboost.load_model(uri)  # XGBClassifier (model_flavour = "xgboost")
    pred_codes = model.predict(test_df)
    pred_labels = [CLASSES[int(code)] for code in pred_codes]

    sub = pd.DataFrame({ID: test_raw[ID], TARGET: pred_labels})
    out = Path("submissions/submission.csv")
    out.parent.mkdir(exist_ok=True)
    sub.to_csv(out, index=False)
    print(f"Saved {len(sub)} rows → {out}  (class mix: {sub[TARGET].value_counts().to_dict()})")


if __name__ == "__main__":
    generate()
