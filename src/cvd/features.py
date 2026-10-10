"""Feature sets and preprocessing for the regression and prediction models."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cvd.data import BINARY_COLS, DIABETES_LEVELS, NUMERIC_COLS, ORDERED_LEVELS, SEX_LEVELS

# Height and weight are left out on purpose: BMI is computed from them, and
# height is mostly a proxy for sex (see the confounding sidebar in notebook 02).
FEATURES_LIFESTYLE = [
    "sex",
    "age_category",
    "bmi",
    "smoking_history",
    "exercise",
    "alcohol_consumption",
    "fruit_consumption",
    "green_vegetables_consumption",
    "fried_potato_consumption",
]
# Variables that may be consequences of heart disease as much as predictors of
# it (self-rated health, doctor visits, comorbid diagnoses).
FEATURES_HEALTH_STATUS = [
    "general_health",
    "checkup",
    "diabetes",
    "arthritis",
    "depression",
    "skin_cancer",
    "other_cancer",
]
FEATURES_ALL = FEATURES_LIFESTYLE + FEATURES_HEALTH_STATUS
FEATURE_SETS = {"all": FEATURES_ALL, "lifestyle": FEATURES_LIFESTYLE}

CATEGORY_LEVELS = {
    **ORDERED_LEVELS,
    **{col: ["No", "Yes"] for col in BINARY_COLS},
    "sex": SEX_LEVELS,
    "diabetes": DIABETES_LEVELS,
}

# Integer codes used by the tree model. Ordered variables keep their natural
# order; diabetes is ordered by severity so a single split can separate
# "any diabetes" from "none".
TREE_CODES = {
    **{col: {lvl: i for i, lvl in enumerate(levels)} for col, levels in ORDERED_LEVELS.items()},
    **{col: {"No": 0, "Yes": 1} for col in BINARY_COLS},
    "sex": {"Female": 0, "Male": 1},
    "diabetes": {"No": 0, "Gestational only": 1, "Pre-diabetes": 2, "Yes": 3},
}

# Reference levels for the explanatory logistic regression.
LOGIT_REFERENCE = {
    "general_health": "Excellent",
    "checkup": "Within the past year",
    "age_category": "18-24",
    "sex": "Female",
    "diabetes": "No",
    **dict.fromkeys(BINARY_COLS, "No"),
}
# Numeric predictors are rescaled so one unit is a meaningful step.
LOGIT_NUMERIC_UNITS = {
    "bmi": (5.0, "BMI (per +5 kg/m²)"),
    "alcohol_consumption": (10.0, "Alcohol (per +10 drinking days/month)"),
    "fruit_consumption": (30.0, "Fruit (per +1 time/day)"),
    "green_vegetables_consumption": (30.0, "Green vegetables (per +1 time/day)"),
    "fried_potato_consumption": (30.0, "Fried potatoes (per +1 time/day)"),
}


def split_feature_types(features: list[str]) -> tuple[list[str], list[str]]:
    """Split a feature list into (categorical, numeric) columns."""
    categorical = [f for f in features if f in CATEGORY_LEVELS]
    numeric = [f for f in features if f in NUMERIC_COLS]
    unknown = set(features) - set(categorical) - set(numeric)
    if unknown:
        raise ValueError(f"Unknown features: {sorted(unknown)}")
    return categorical, numeric


class TreeEncoder(BaseEstimator, TransformerMixin):
    """Encode the cleaned data frame as one numeric column per feature.

    Keeping one column per original variable makes SHAP values directly
    readable ("age group contributed +0.8") instead of spreading a variable
    over many dummy columns.
    """

    def __init__(self, features: list[str]):
        self.features = features

    def fit(self, X: pd.DataFrame, y=None):
        split_feature_types(self.features)
        self.feature_names_in_ = list(self.features)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for col in self.features:
            if col in TREE_CODES:
                codes = X[col].astype("object").map(TREE_CODES[col])
                if codes.isna().any():
                    bad = sorted(X.loc[codes.isna(), col].astype(str).unique())
                    raise ValueError(f"Unexpected values in {col!r}: {bad}")
                out[col] = codes.astype("int8")
            else:
                out[col] = X[col].astype("float64")
        return out

    def get_feature_names_out(self, input_features=None):
        return list(self.features)


def make_linear_preprocessor(features: list[str]) -> ColumnTransformer:
    """One-hot encode categoricals (first level dropped) and standardise numerics."""
    categorical, numeric = split_feature_types(features)
    return ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(
                    categories=[CATEGORY_LEVELS[c] for c in categorical],
                    drop="first",
                    sparse_output=False,
                ),
                categorical,
            ),
            ("num", StandardScaler(), numeric),
        ],
        verbose_feature_names_out=False,
    )


def logit_design_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Design matrix for the explanatory logistic regression.

    Categorical variables are dummy-coded against ``LOGIT_REFERENCE``; numeric
    variables are divided by the step in ``LOGIT_NUMERIC_UNITS`` so that odds
    ratios refer to a meaningful change. Column names are human-readable.
    """
    categorical, numeric = split_feature_types(features)
    parts = []
    for col in features:
        if col in categorical:
            reference = LOGIT_REFERENCE[col]
            levels = [lvl for lvl in CATEGORY_LEVELS[col] if lvl != reference]
            values = df[col].astype("object")
            block = pd.DataFrame(
                {f"{col}: {lvl}": (values == lvl).astype("float64") for lvl in levels},
                index=df.index,
            )
            parts.append(block)
        else:
            step, label = LOGIT_NUMERIC_UNITS[col]
            parts.append((df[col] / step).rename(label).to_frame())
    design = pd.concat(parts, axis=1)
    design.insert(0, "Intercept", 1.0)
    return design
