"""Evaluation for student-health — multiclass classification.

Scores the model on a held-out validation split using the same seed as
training so the val partition is consistent across runs.
Reports accuracy, macro-f1, and macro roc_auc (one-vs-rest).
"""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from kitchen.modeling import classification_metrics, train_val_split
from kitchen.steps import Evaluator
from kitchen.store import DataStore


class StudentHealthEvaluator(Evaluator):
    """Multiclass classification evaluator.

    Overrides run() to stash params as an instance attribute so that
    evaluate() can access the target column and random seed — the base class
    does not forward params to evaluate().
    """

    def run(self, model: object, store: DataStore, params: dict) -> dict[str, float]:
        self._params = params  # stash so evaluate() can read target + seed
        return super().run(model, store, params)

    def evaluate(self, model: object, df: pd.DataFrame) -> dict[str, float]:
        params = self._params
        target = params["model"]["target"]
        seed = params["model"].get("random_state", 42)

        _, val_df = train_val_split(df, target_col=target, seed=seed)
        features = [c for c in val_df.columns if c != target]
        X_val, y_val = val_df[features], val_df[target]

        y_pred = model.predict(X_val)
        y_proba = (
            model.predict_proba(X_val) if hasattr(model, "predict_proba") else None
        )
        metrics = classification_metrics(y_val, y_pred, y_proba=y_proba, average="macro")
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_val, y_pred))
        return metrics


def evaluate(model: object, params: dict, store: DataStore) -> dict[str, float]:
    return StudentHealthEvaluator().run(model, store, params)
