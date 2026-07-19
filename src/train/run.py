"""Model training for student-health — multiclass (XGBoost baseline, balanced-accuracy aligned).

Fits an XGBClassifier (multi:softprob) on the engineered features. Two things matter for this
competition's metric (balanced accuracy) and its 86/8/6 class imbalance:

* ``sample_weight`` uses sklearn's 'balanced' weights so the majority ``at-risk`` class does not
  swamp training — without it the model collapses onto one class and scores ~0.33 (chance);
* ``enable_categorical=True`` lets XGBoost consume the pandas ``category`` feature columns directly.

Logs the standard metrics plus ``val_balanced_accuracy`` (the leaderboard metric, and the primary
metric for the leaderboard + ``--auto-promote``) to the active MLflow run.
"""
from __future__ import annotations

import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

from kitchen.modeling import classification_metrics, train_val_split
from kitchen.steps import Trainer
from kitchen.store import DataStore
from kitchen.tracking import Tracker


class StudentHealthTrainer(Trainer):
    model_flavour = "xgboost"

    def fit(self, df: pd.DataFrame, params: dict) -> xgb.XGBClassifier:
        target = params["model"]["target"]
        seed = params["model"].get("random_state", 42)

        train_df, val_df = train_val_split(df, target_col=target, seed=seed)
        features = [c for c in df.columns if c != target]
        X_train, y_train = train_df[features], train_df[target]
        X_val, y_val = val_df[features], val_df[target]

        p = params["model"].get("xgb", {})
        model = xgb.XGBClassifier(
            objective="multi:softprob",
            enable_categorical=True,
            n_estimators=p.get("n_estimators", 300),
            max_depth=p.get("max_depth", 6),
            learning_rate=p.get("learning_rate", 0.05),
            subsample=p.get("subsample", 0.8),
            colsample_bytree=p.get("colsample_bytree", 0.8),
            random_state=seed,
            eval_metric="mlogloss",
        )
        sample_weight = compute_sample_weight("balanced", y_train)
        model.fit(X_train, y_train, sample_weight=sample_weight)

        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)  # full probability matrix for roc_auc (ovr)
        val_metrics = classification_metrics(y_val, y_pred, y_proba=y_proba, average="macro")
        val_metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_val, y_pred))
        mlflow.log_metrics({"val_" + k: v for k, v in val_metrics.items()})
        return model


def train(params: dict, store: DataStore, tracker: Tracker) -> object:
    return StudentHealthTrainer().run(store, tracker, params)
