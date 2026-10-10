"""Model training, cross-validation, evaluation and persistence."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from cvd import SEED
from cvd.data import LABELS, PROJECT_ROOT, TARGET
from cvd.features import TreeEncoder, logit_design_matrix, make_linear_preprocessor

# Small enough (under 1 MB) to version, so the app can be deployed without retraining.
MODELS_DIR = PROJECT_ROOT / "app" / "model"

LGBM_DEFAULTS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 200,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
}


# --------------------------------------------------------------------------
# Explanatory logistic regression (statsmodels)
# --------------------------------------------------------------------------
def fit_logit(df: pd.DataFrame, features: list[str], target: str = TARGET):
    """Fit a statsmodels logistic regression on the design matrix of ``features``."""
    import statsmodels.api as sm  # imported here so the app does not need statsmodels

    design = logit_design_matrix(df, features)
    return sm.Logit(df[target].astype(float), design).fit(disp=False, maxiter=200)


def odds_ratio_table(result, alpha: float = 0.05) -> pd.DataFrame:
    """Odds ratios with confidence intervals from a fitted statsmodels Logit."""
    ci = result.conf_int(alpha=alpha)
    table = pd.DataFrame(
        {
            "odds_ratio": np.exp(result.params),
            "ci_low": np.exp(ci[0]),
            "ci_high": np.exp(ci[1]),
            "p_value": result.pvalues,
        }
    )
    return table.drop(index="Intercept")


def unadjusted_odds_ratios(
    df: pd.DataFrame, features: list[str], target: str = TARGET
) -> pd.DataFrame:
    """Odds ratios from one single-predictor logistic regression per feature."""
    return pd.concat([odds_ratio_table(fit_logit(df, [f], target)) for f in features])


def logit_fit_summary(results: dict) -> pd.DataFrame:
    """Fit statistics for several fitted Logit models, one row per model."""
    rows = {
        name: {
            "n_obs": int(res.nobs),
            "n_params": len(res.params),
            "log_likelihood": res.llf,
            "aic": res.aic,
            "mcfadden_r2": res.prsquared,
            "converged": bool(res.mle_retvals["converged"]),
        }
        for name, res in results.items()
    }
    return pd.DataFrame(rows).T


def likelihood_ratio_test(restricted, full) -> dict[str, float]:
    """Likelihood-ratio test of a nested (restricted) Logit against the full model."""
    from scipy import stats as sps

    statistic = 2 * (full.llf - restricted.llf)
    dof = len(full.params) - len(restricted.params)
    return {
        "lr_stat": float(statistic),
        "dof": int(dof),
        "p_value": float(sps.chi2.sf(statistic, dof)),
    }


# --------------------------------------------------------------------------
# Predictive pipelines (scikit-learn)
# --------------------------------------------------------------------------
def train_test(df: pd.DataFrame, test_size: float = 0.2, seed: int = SEED):
    """Stratified train/test split of the cleaned data frame."""
    return train_test_split(df, test_size=test_size, stratify=df[TARGET], random_state=seed)


def make_dummy_pipeline() -> Pipeline:
    """Majority-class baseline: predicts the training prevalence for everyone."""
    return Pipeline([("model", DummyClassifier(strategy="prior"))])


def make_logreg_pipeline(features: list[str], class_weight=None) -> Pipeline:
    return Pipeline(
        [
            ("prep", make_linear_preprocessor(features)),
            ("model", LogisticRegression(max_iter=1000, class_weight=class_weight)),
        ]
    )


def make_lgbm_pipeline(features: list[str], seed: int = SEED, **params) -> Pipeline:
    settings = {**LGBM_DEFAULTS, **params}
    return Pipeline(
        [
            ("prep", TreeEncoder(features)),
            (
                "model",
                LGBMClassifier(
                    **settings,
                    random_state=seed,
                    n_jobs=4,
                    deterministic=True,
                    force_row_wise=True,
                    verbose=-1,
                ),
            ),
        ]
    )


def cv_splitter(n_splits: int = 5, seed: int = SEED) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def cross_val_oof(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, n_splits: int = 5, seed: int = SEED
) -> tuple[np.ndarray, pd.DataFrame]:
    """Stratified k-fold CV. Returns out-of-fold probabilities and per-fold metrics."""
    y_arr = np.asarray(y)
    oof = np.zeros(len(y_arr), dtype=float)
    rows = []
    for fold, (train_idx, valid_idx) in enumerate(cv_splitter(n_splits, seed).split(X, y_arr)):
        model = clone(pipeline).fit(X.iloc[train_idx], y_arr[train_idx])
        oof[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]
        rows.append({"fold": fold, **probability_metrics(y_arr[valid_idx], oof[valid_idx])})
    return oof, pd.DataFrame(rows).set_index("fold")


def probability_metrics(y_true, y_prob) -> dict[str, float]:
    """Threshold-free metrics. PR-AUC is average precision."""
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def threshold_metrics(y_true, y_prob, threshold: float) -> dict[str, float]:
    """Precision, recall and share flagged when predicting positive at ``threshold``."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "flagged_share": float(y_pred.mean()),
    }


def best_fbeta_threshold(y_true, y_prob, beta: float = 2.0) -> float:
    """Threshold maximising F-beta (beta > 1 weights recall above precision)."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    precision, recall = precision[:-1], recall[:-1]
    denom = beta**2 * precision + recall
    fbeta = np.where(
        denom > 0, (1 + beta**2) * precision * recall / np.where(denom > 0, denom, 1), 0
    )
    return float(thresholds[int(np.argmax(fbeta))])


def threshold_table(y_true, y_prob, recalls: list[float]) -> pd.DataFrame:
    """Operating points: for each target recall, the highest threshold that reaches it."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    rows = []
    for target in recalls:
        idx = np.where(recall[:-1] >= target)[0][-1]
        rows.append({"target_recall": target, **threshold_metrics(y_true, y_prob, thresholds[idx])})
    return pd.DataFrame(rows).set_index("target_recall")


def bootstrap_metric_ci(
    y_true, y_prob, metric, n_boot: int = 500, alpha: float = 0.05, seed: int = SEED
) -> tuple[float, float]:
    """Percentile bootstrap CI of a metric(y_true, y_prob) over resampled rows."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        estimates[i] = metric(y_true[idx], y_prob[idx])
    low, high = np.quantile(estimates, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def shap_values(pipeline: Pipeline, X: pd.DataFrame):
    """SHAP explanation (log-odds scale) for a fitted LightGBM pipeline.

    Returns a ``shap.Explanation`` whose ``data`` is the encoded feature matrix and
    whose ``display_data`` holds the original, readable values.
    """
    import shap

    encoder = pipeline.named_steps["prep"]
    encoded = encoder.transform(X)
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    raw = explainer(encoded)
    values, base = raw.values, np.asarray(raw.base_values)
    if values.ndim == 3:  # some versions return one slice per class
        values, base = values[:, :, 1], base[:, 1]
    return shap.Explanation(
        values=values,
        base_values=base,
        data=encoded.to_numpy(dtype=float),
        display_data=display_values(X[encoder.features]),
        feature_names=[LABELS[f] for f in encoder.features],
    )


def tree_contributions(pipeline: Pipeline, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Per-feature contributions (log-odds) from LightGBM's built-in TreeSHAP.

    Gives the same values as :func:`shap_values` without needing the ``shap``
    package, which keeps the app's dependencies small. Returns a data frame of
    contributions (one row per row of ``X``, one column per feature) and the
    base value for each row.
    """
    encoder = pipeline.named_steps["prep"]
    booster = pipeline.named_steps["model"].booster_
    raw = booster.predict(encoder.transform(X), pred_contrib=True)
    contributions = pd.DataFrame(raw[:, :-1], index=X.index, columns=list(encoder.features))
    return contributions, raw[:, -1]


def display_values(X: pd.DataFrame) -> np.ndarray:
    """Original labels (not integer codes) for showing next to SHAP values."""
    shown = X.astype(object).copy()
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            shown[col] = X[col].map(lambda v: f"{v:g}")
    return shown.to_numpy(dtype=object)


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
def save_bundle(bundle: dict, name: str = "heart_disease_models.joblib") -> Path:
    """Save fitted pipelines plus the metadata the app needs."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / name
    joblib.dump(bundle, path)
    return path


def load_bundle(name: str = "heart_disease_models.joblib") -> dict:
    path = MODELS_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `make notebooks` to train and save the models."
        )
    return joblib.load(path)
