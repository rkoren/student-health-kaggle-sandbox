"""Challenger experiment for student-health.

Extend the baseline: add features, tune hyperparams, or swap the model.
Tag: model_variant=challenger.

Usage:
    python experiments/challenger.py
"""
from __future__ import annotations

import yaml
from dotenv import load_dotenv

load_dotenv()

from experiments.baseline import run_variant

VARIANT = "challenger"


def challenger(params_file: str = "menu.yaml") -> None:
    with open(params_file) as f:
        params = yaml.safe_load(f)

    # Challenger: 5-fold CV for an honest OOF metric, champion refit on 100% of rows.
    # 300 fixed rounds won the OOF checkpoint sweep (0.94981); early stopping on
    # weighted mlogloss quits ~90 rounds in and scores worse, so leave it off.
    params["model"]["cv_folds"] = 5
    params["model"]["xgb"].update(n_estimators=300)

    run_variant(params, VARIANT)


if __name__ == "__main__":
    challenger()
