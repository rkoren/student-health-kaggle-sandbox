"""Tests for student-health feature engineering."""
import pandas as pd
import pytest

from src.features.run import (
    CATEGORICALS,
    CLASSES,
    FEATURES,
    NUMERIC,
    TARGET,
    StudentHealthFeatures,
    engineer,
)


@pytest.fixture
def raw_row() -> pd.DataFrame:
    """One representative raw training row (numeric metrics + categoricals + target)."""
    return pd.DataFrame([{
        "id": 1,
        "health_condition": "unhealthy",
        "sleep_duration": 6.5,
        "heart_rate": 72.0,
        "bmi": 24.1,
        "calorie_expenditure": 2200.0,
        "step_count": 8000.0,
        "exercise_duration": 30.0,
        "water_intake": 2.0,
        "diet_type": "veg",
        "stress_level": "medium",
        "sleep_quality": "good",
        "physical_activity_level": "active",
        "smoking_alcohol": "no",
        "gender": "female",
    }])


def test_engineer_returns_feature_columns_with_category_dtype(raw_row):
    feats = engineer(raw_row)
    assert list(feats.columns) == FEATURES
    for col in CATEGORICALS:
        assert isinstance(feats[col].dtype, pd.CategoricalDtype)
    for col in NUMERIC:
        assert pd.api.types.is_numeric_dtype(feats[col])


def test_build_attaches_label_encoded_target(raw_row):
    out = StudentHealthFeatures().build(raw_row, params={})
    assert TARGET in out.columns
    # "unhealthy" is index 2 in the fixed CLASSES order.
    assert out[TARGET].iloc[0] == CLASSES.index("unhealthy") == 2
    assert out[TARGET].dtype.kind == "i"


def test_engineer_maps_unseen_category_to_missing():
    df = pd.DataFrame([{**{n: 0.0 for n in NUMERIC}, **{c: "??" for c in CATEGORICALS}}])
    feats = engineer(df)
    # An out-of-vocabulary categorical value becomes NaN (a missing category), never an error.
    for col in CATEGORICALS:
        assert feats[col].isna().all()


def test_features_list_excludes_id_and_target():
    assert "id" not in FEATURES and TARGET not in FEATURES
    assert FEATURES == NUMERIC + list(CATEGORICALS)
