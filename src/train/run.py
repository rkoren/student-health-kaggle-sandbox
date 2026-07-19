"""Model training for student-health — multiclass (XGBoost baseline, balanced-accuracy aligned).

Fits an XGBClassifier (multi:softprob) on the engineered features. Two things matter for this
competition's metric (balanced accuracy) and its 86/8/6 class imbalance:

* ``sample_weight`` uses sklearn's 'balanced' weights so the majority ``at-risk`` class does not
  swamp training — without it the model collapses onto one class and scores ~0.33 (chance);
* ``enable_categorical=True`` lets XGBoost consume the pandas ``category`` feature columns directly.

Two training modes, chosen by ``model.cv_folds``:

* unset (default) — single 80/20 split, val metrics from the held-out 20%;
* set to k — k-fold StratifiedKFold: val metrics come from out-of-fold predictions (honest,
  every row scored by a model that never saw it), then the returned model is refit on ALL rows
  with the median early-stopped round count, so the champion sees 100% of the data.

Logs the standard metrics plus ``val_balanced_accuracy`` (the leaderboard metric, and the primary
metric for the leaderboard + ``--auto-promote``) to the active MLflow run.
"""
from __future__ import annotations

import numpy as np
import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
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
        p = params["model"].get("xgb", {})
        cv_folds = params["model"].get("cv_folds")

        features = [c for c in df.columns if c != target]
        # val_* metrics from different validation schemes aren't comparable (OOF is
        # honest, a single split is optimistic) — tag the run so promotion tooling
        # and humans compare like with like.
        mlflow.set_tag("validation_scheme", f"oof-{int(cv_folds)}fold" if cv_folds else "single-split")
        if cv_folds:
            model, y_val, y_pred, y_proba = self._fit_cv(
                df[features], df[target], p, seed, int(cv_folds)
            )
        else:
            model, y_val, y_pred, y_proba = self._fit_single(df, features, target, p, seed)

        val_metrics = classification_metrics(y_val, y_pred, y_proba=y_proba, average="macro")
        val_metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_val, y_pred))
        mlflow.log_metrics({"val_" + k: v for k, v in val_metrics.items()})
        return model

    def _make_model(
        self, p: dict, seed: int, n_estimators: int | None = None, early_stopping: int | None = None
    ) -> xgb.XGBClassifier:
        return xgb.XGBClassifier(
            objective="multi:softprob",
            enable_categorical=True,
            n_estimators=n_estimators or p.get("n_estimators", 300),
            max_depth=p.get("max_depth", 6),
            learning_rate=p.get("learning_rate", 0.05),
            subsample=p.get("subsample", 0.8),
            colsample_bytree=p.get("colsample_bytree", 0.8),
            min_child_weight=p.get("min_child_weight", 1),
            reg_alpha=p.get("reg_alpha", 0),
            reg_lambda=p.get("reg_lambda", 1),
            random_state=seed,
            eval_metric="mlogloss",
            early_stopping_rounds=early_stopping,
        )

    def _fit_single(self, df, features, target, p, seed):
        train_df, val_df = train_val_split(df, target_col=target, seed=seed)
        X_train, y_train = train_df[features], train_df[target]
        X_val, y_val = val_df[features], val_df[target]

        early_stopping = p.get("early_stopping_rounds")
        model = self._make_model(p, seed, early_stopping=early_stopping)
        fit_kwargs: dict = {"sample_weight": compute_sample_weight("balanced", y_train)}
        if early_stopping:
            # Weight the val split the same way so early stopping optimizes the
            # balanced objective, not raw mlogloss on the 86%-majority class.
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["sample_weight_eval_set"] = [compute_sample_weight("balanced", y_val)]
            fit_kwargs["verbose"] = False
        model.fit(X_train, y_train, **fit_kwargs)
        if early_stopping:
            mlflow.log_metric("best_iteration", model.best_iteration)
        return model, y_val, model.predict(X_val), model.predict_proba(X_val)

    def _fit_cv(self, X, y, p, seed, n_folds):
        """OOF metrics from k folds, then a final full-data refit for the returned model.

        With ``early_stopping_rounds`` unset the folds train the full fixed round count —
        weighted-mlogloss early stopping quits ~90 rounds in and costs ~0.0004 OOF balanced
        accuracy versus the 300-round optimum found by the checkpoint sweep.
        """
        early_stopping = p.get("early_stopping_rounds")
        oof = np.zeros((len(y), y.nunique()))
        best_rounds = []
        for tr_i, va_i in StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(X, y):
            m = self._make_model(p, seed, early_stopping=early_stopping)
            kwargs: dict = {"sample_weight": compute_sample_weight("balanced", y.iloc[tr_i])}
            if early_stopping:
                kwargs["eval_set"] = [(X.iloc[va_i], y.iloc[va_i])]
                kwargs["sample_weight_eval_set"] = [compute_sample_weight("balanced", y.iloc[va_i])]
                kwargs["verbose"] = False
            m.fit(X.iloc[tr_i], y.iloc[tr_i], **kwargs)
            oof[va_i] = m.predict_proba(X.iloc[va_i])
            best_rounds.append((m.best_iteration + 1) if early_stopping else m.n_estimators)

        final_rounds = int(np.median(best_rounds))
        mlflow.log_metric("best_iteration", final_rounds)
        # Refit on 100% of rows: no early stopping possible, so reuse the CV round count.
        model = self._make_model(p, seed, n_estimators=final_rounds)
        model.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
        return model, y, oof.argmax(1), oof


def train(params: dict, store: DataStore, tracker: Tracker) -> object:
    return StudentHealthTrainer().run(store, tracker, params)
