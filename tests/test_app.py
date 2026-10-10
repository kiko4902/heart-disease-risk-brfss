"""Smoke test of the Streamlit app against a small model bundle trained on synthetic data."""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from cvd import features, models
from cvd.data import NUMERIC_COLS, TARGET

APP = str(Path(__file__).resolve().parents[1] / "app" / "app.py")


@pytest.fixture(autouse=True)
def fresh_resource_cache():
    """The app caches the loaded bundle; clear it so each test sees its own models dir."""
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


@pytest.fixture
def synthetic_bundle(df, tmp_path, monkeypatch):
    monkeypatch.setattr(models, "MODELS_DIR", tmp_path)
    fitted = {
        name: models.make_lgbm_pipeline(feats, n_estimators=20, min_child_samples=10).fit(
            df[feats], df[TARGET]
        )
        for name, feats in features.FEATURE_SETS.items()
    }
    metrics = {"pr_auc": 0.3, "roc_auc": 0.8, "brier": 0.06}
    models.save_bundle(
        {
            "models": fitted,
            "features": dict(features.FEATURE_SETS),
            "thresholds": dict.fromkeys(fitted, 0.1),
            "base_rate": float(df[TARGET].mean()),
            "test_metrics": dict.fromkeys(fitted, metrics),
            "numeric_summary": df[NUMERIC_COLS].describe().loc[["min", "50%", "max"]].to_dict(),
            "n_train": len(df),
            "seed": 42,
        }
    )


def test_app_renders_and_estimates(synthetic_bundle):
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    assert "not medical advice" in app.warning[0].value
    assert len(app.metric) == 0  # nothing shown before the form is submitted

    app.button[0].click().run()
    assert not app.exception
    assert [m.label for m in app.metric] == [
        "Estimated probability",
        "Average in the survey",
        "Relative to average",
    ]
    assert app.metric[0].value.endswith("%")


def test_app_lifestyle_model_has_no_health_status_inputs(synthetic_bundle):
    app = AppTest.from_file(APP, default_timeout=60).run()
    labels_all = {w.label for w in app.selectbox}
    app.sidebar.radio[0].set_value("lifestyle").run()
    labels_lifestyle = {w.label for w in app.selectbox}
    assert "Diabetes" in labels_all
    assert "Diabetes" not in labels_lifestyle
    app.button[0].click().run()
    assert not app.exception
    assert len(app.metric) == 3


def test_app_reports_missing_models(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "MODELS_DIR", tmp_path)
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    assert "make notebooks" in app.error[0].value
