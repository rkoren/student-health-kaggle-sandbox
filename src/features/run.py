"""Feature engineering for student-health (Playground S6E7 — predict health_condition).

The competition inputs are 7 numeric health metrics + 7 low-cardinality categorical columns,
all with missing values. The baseline keeps transformation deliberately light:

* numeric columns pass through untouched — XGBoost handles NaN natively;
* categorical columns become pandas ``category`` dtype with an explicit, fixed category list so
  the encoding is *identical* on train and test (XGBoost consumes them directly via
  ``enable_categorical=True``); a NaN stays a missing category.

The target ``health_condition`` is label-encoded to 0/1/2 (XGBoost rejects string labels) using
the fixed ``CLASSES`` order; ``flows/generate_submission.py`` decodes predictions back to strings.
"""
from __future__ import annotations

import pandas as pd
from kitchen.steps import FeatureBuilder
from kitchen.store import DataStore

ID = "id"
TARGET = "health_condition"

# Fixed target label order → integer codes 0, 1, 2 (decoded back in generate_submission).
CLASSES: list[str] = ["at-risk", "fit", "unhealthy"]

# Numeric health metrics — passed through as-is (XGBoost handles the missing values).
NUMERIC: list[str] = [
    "sleep_duration",
    "heart_rate",
    "bmi",
    "calorie_expenditure",
    "step_count",
    "exercise_duration",
    "water_intake",
]

# Categorical columns → explicit category orderings, applied identically to train and test.
CATEGORICALS: dict[str, list[str]] = {
    "diet_type": ["balanced", "non-veg", "veg"],
    "stress_level": ["high", "low", "medium"],
    "sleep_quality": ["average", "good", "poor"],
    "physical_activity_level": ["active", "moderate", "sedentary"],
    "smoking_alcohol": ["no", "occasional", "yes"],
    "gender": ["female", "male", "other"],
}

# Every column handed to the model (stable order; excludes id and target).
FEATURES: list[str] = NUMERIC + list(CATEGORICALS)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model-ready feature matrix (``FEATURES`` columns) for train *or* test.

    Categoricals get a fixed ``category`` dtype so both splits encode identically; numerics
    pass through. No target column here — callers add it when the target is present.
    """
    out = df[NUMERIC].copy()
    for col, cats in CATEGORICALS.items():
        out[col] = pd.Categorical(df[col], categories=cats)
    return out[FEATURES]


class StudentHealthFeatures(FeatureBuilder):
    def build(self, raw: pd.DataFrame | dict[str, pd.DataFrame], params: dict) -> pd.DataFrame:
        """Engineer features and attach the label-encoded target (train.py separates it out)."""
        feats = engineer(raw)
        feats[TARGET] = raw[TARGET].map({c: i for i, c in enumerate(CLASSES)}).astype("int64")
        return feats


def build(params: dict, store: DataStore) -> None:
    StudentHealthFeatures().run(store, params)
