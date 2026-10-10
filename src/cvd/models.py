"""Model training, cross-validation, evaluation and persistence."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
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
from cvd.data import PROJECT_ROOT, TARGET
from cvd.features import TreeEncoder, logit_design_matrix, make_linear_preprocessor

MODELS_DIR = PROJECT_ROOT / "models"

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


def shap_values(pipeline: Pipeline, X: pd.DataFrame):
    """SHAP explanation (log-odds scale) for a fitted LightGBM pipeline.

    Returns a ``shap.Explanation`` whose ``data`` is the encoded feature matrix.
    """
    import shap

    encoded = pipeline.named_steps["prep"].transform(X)
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    explanation = explainer(encoded)
    if explanation.values.ndim == 3:  # some versions return one slice per class
        explanation = explanation[:, :, 1]
    return explanation


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
def save_bundle(bundle: dict, name: str = "heart_disease_models.joblib") -> Path:
    """Save fitted pipelines plus the metadata the app needs."""
    MODELS_DIR.mkdir(exist_ok=True)
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
