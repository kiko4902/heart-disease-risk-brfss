import numpy as np
import pandas as pd
import pytest

from cvd import features, models
from cvd.data import TARGET


def test_feature_sets_are_consistent(df):
    assert set(features.FEATURES_LIFESTYLE) < set(features.FEATURES_ALL)
    assert TARGET not in features.FEATURES_ALL
    assert set(features.FEATURES_ALL) <= set(df.columns)
    # deliberately excluded, see features.py
    assert "height_cm" not in features.FEATURES_ALL
    assert "weight_kg" not in features.FEATURES_ALL


@pytest.mark.parametrize("name", ["all", "lifestyle"])
def test_tree_encoder_shape_and_dtypes(df, name):
    feats = features.FEATURE_SETS[name]
    encoded = features.TreeEncoder(feats).fit_transform(df)
    assert encoded.shape == (len(df), len(feats))
    assert list(encoded.columns) == feats
    assert all(pd.api.types.is_numeric_dtype(t) for t in encoded.dtypes)
    assert not encoded.isna().any().any()


def test_tree_encoder_codes_follow_order(df):
    encoded = features.TreeEncoder(["age_category", "sex", "diabetes"]).fit_transform(df)
    assert encoded.loc[df["age_category"] == "18-24", "age_category"].eq(0).all()
    assert encoded.loc[df["age_category"] == "80+", "age_category"].eq(12).all()
    assert encoded.loc[df["sex"] == "Male", "sex"].eq(1).all()
    assert encoded.loc[df["diabetes"] == "Yes", "diabetes"].eq(3).all()


def test_tree_encoder_accepts_plain_strings_and_rejects_unknown(df):
    row = df.iloc[[0]][features.FEATURES_ALL].astype({"sex": "object"})
    assert features.TreeEncoder(features.FEATURES_ALL).fit(df).transform(row).shape == (1, 16)
    row["sex"] = "Other"
    with pytest.raises(ValueError, match="sex"):
        features.TreeEncoder(features.FEATURES_ALL).fit(df).transform(row)


def test_linear_preprocessor_shape(df):
    prep = features.make_linear_preprocessor(features.FEATURES_ALL)
    matrix = prep.fit_transform(df)
    categorical, numeric = features.split_feature_types(features.FEATURES_ALL)
    expected = sum(len(features.CATEGORY_LEVELS[c]) - 1 for c in categorical) + len(numeric)
    assert matrix.shape == (len(df), expected)
    assert np.isfinite(matrix).all()


def test_logit_design_matrix_drops_reference_levels(df):
    design = features.logit_design_matrix(df, features.FEATURES_ALL)
    assert design.columns[0] == "Intercept"
    assert "sex: Male" in design.columns
    assert "sex: Female" not in design.columns
    assert "general_health: Excellent" not in design.columns
    assert "general_health: Poor" in design.columns
    assert design["BMI (per +5 kg/m²)"].equals(df["bmi"] / 5.0)
    assert np.linalg.matrix_rank(design.to_numpy()) == design.shape[1]


def test_split_feature_types_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown features"):
        features.split_feature_types(["bmi", "shoe_size"])


def test_train_test_split_is_stratified_and_seeded(df):
    train, test = models.train_test(df)
    assert len(train) + len(test) == len(df)
    assert abs(train[TARGET].mean() - test[TARGET].mean()) < 0.02
    assert models.train_test(df)[1].index.equals(test.index)


@pytest.mark.parametrize(
    "factory", [models.make_logreg_pipeline, models.make_lgbm_pipeline], ids=["logreg", "lgbm"]
)
def test_pipelines_fit_and_predict_probabilities(df, factory):
    feats = features.FEATURES_LIFESTYLE
    kwargs = (
        {"n_estimators": 20, "min_child_samples": 10}
        if factory is models.make_lgbm_pipeline
        else {}
    )
    pipe = factory(feats, **kwargs).fit(df[feats], df[TARGET])
    prob = pipe.predict_proba(df[feats])
    assert prob.shape == (len(df), 2)
    assert ((prob >= 0) & (prob <= 1)).all()


def test_cross_val_oof_shapes_and_reproducibility(df):
    feats = features.FEATURES_LIFESTYLE
    pipe = models.make_logreg_pipeline(feats)
    oof, folds = models.cross_val_oof(pipe, df[feats], df[TARGET], n_splits=3)
    assert oof.shape == (len(df),)
    assert folds.shape == (3, 3)
    oof_again, _ = models.cross_val_oof(pipe, df[feats], df[TARGET], n_splits=3)
    assert np.allclose(oof, oof_again)
    assert folds["roc_auc"].mean() > 0.55  # the synthetic target depends on age


def test_dummy_pipeline_predicts_prevalence(df):
    pipe = models.make_dummy_pipeline().fit(df[features.FEATURES_LIFESTYLE], df[TARGET])
    prob = pipe.predict_proba(df[features.FEATURES_LIFESTYLE])[:, 1]
    assert np.allclose(prob, df[TARGET].mean())


def test_metrics_on_known_inputs():
    y = np.array([0, 0, 1, 1])
    perfect = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = models.probability_metrics(y, perfect)
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["brier"] == pytest.approx(np.mean([0.01, 0.04, 0.04, 0.01]))
    at_half = models.threshold_metrics(y, np.array([0.6, 0.2, 0.8, 0.4]), 0.5)
    assert at_half["precision"] == pytest.approx(0.5)
    assert at_half["recall"] == pytest.approx(0.5)
    assert at_half["flagged_share"] == pytest.approx(0.5)


def test_best_fbeta_threshold_prefers_recall_for_large_beta():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.1).astype(int)
    prob = np.clip(0.1 + 0.25 * y + rng.normal(0, 0.1, 2000), 0, 1)
    t_recall = models.best_fbeta_threshold(y, prob, beta=2)
    t_precision = models.best_fbeta_threshold(y, prob, beta=0.5)
    assert t_recall < t_precision
    assert (
        models.threshold_metrics(y, prob, t_recall)["recall"]
        > models.threshold_metrics(y, prob, t_precision)["recall"]
    )


def test_fit_logit_and_odds_ratio_table(df):
    result = models.fit_logit(df, ["sex", "bmi", "age_category"])
    table = models.odds_ratio_table(result)
    assert "Intercept" not in table.index
    assert (table["ci_low"] < table["odds_ratio"]).all()
    assert (table["odds_ratio"] < table["ci_high"]).all()
    assert table.loc["age_category: 80+", "odds_ratio"] > 1


def test_shap_values_are_additive(df):
    feats = features.FEATURES_LIFESTYLE
    pipe = models.make_lgbm_pipeline(feats, n_estimators=20, min_child_samples=10)
    pipe.fit(df[feats], df[TARGET])
    sample = df[feats].iloc[:50]
    explanation = models.shap_values(pipe, sample)
    assert explanation.values.shape == (50, len(feats))
    log_odds = explanation.values.sum(axis=1) + np.ravel(explanation.base_values)
    prob = 1 / (1 + np.exp(-log_odds))
    assert np.allclose(prob, pipe.predict_proba(sample)[:, 1], atol=1e-6)


def test_bundle_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "MODELS_DIR", tmp_path)
    models.save_bundle({"a": 1}, "x.joblib")
    assert models.load_bundle("x.joblib") == {"a": 1}
    with pytest.raises(FileNotFoundError, match="make notebooks"):
        models.load_bundle("missing.joblib")


def test_shap_explanation_shows_readable_values(df):
    feats = features.FEATURES_LIFESTYLE
    pipe = models.make_lgbm_pipeline(feats, n_estimators=10, min_child_samples=10)
    pipe.fit(df[feats], df[TARGET])
    explanation = models.shap_values(pipe, df[feats].iloc[:5])
    assert explanation.feature_names[0] == "Sex"
    assert explanation.display_data[0, 0] in {"Female", "Male"}
    assert explanation.data.dtype == float


def test_threshold_table_reaches_target_recall():
    rng = np.random.default_rng(0)
    y = (rng.random(3000) < 0.1).astype(int)
    prob = np.clip(0.1 + 0.2 * y + rng.normal(0, 0.1, 3000), 0, 1)
    table = models.threshold_table(y, prob, [0.5, 0.8])
    assert (table["recall"] >= table.index.to_numpy()).all()
    assert table.loc[0.8, "threshold"] < table.loc[0.5, "threshold"]
    assert table.loc[0.8, "flagged_share"] > table.loc[0.5, "flagged_share"]


def test_bootstrap_metric_ci_is_seeded():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.2).astype(int)
    prob = np.clip(0.2 + 0.3 * y + rng.normal(0, 0.15, 500), 0, 1)
    metric = lambda a, b: models.probability_metrics(a, b)["pr_auc"]  # noqa: E731
    first = models.bootstrap_metric_ci(y, prob, metric, n_boot=50)
    assert first == models.bootstrap_metric_ci(y, prob, metric, n_boot=50)
    assert first[0] < metric(y, prob) < first[1]


def test_tree_contributions_match_shap_and_add_up(df):
    feats = features.FEATURES_ALL
    pipe = models.make_lgbm_pipeline(feats, n_estimators=20, min_child_samples=10)
    pipe.fit(df[feats], df[TARGET])
    sample = df[feats].iloc[:40]
    contributions, base = models.tree_contributions(pipe, sample)
    assert list(contributions.columns) == feats
    assert np.allclose(contributions.to_numpy(), models.shap_values(pipe, sample).values, atol=1e-6)
    prob = 1 / (1 + np.exp(-(contributions.sum(axis=1).to_numpy() + base)))
    assert np.allclose(prob, pipe.predict_proba(sample)[:, 1], atol=1e-6)
