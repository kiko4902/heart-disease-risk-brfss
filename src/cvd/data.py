"""Loading and cleaning of the CVD (BRFSS 2021) dataset."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = Path(os.environ.get("CVD_DATA_PATH", PROJECT_ROOT / "data" / "CVD_cleaned.csv"))

TARGET = "heart_disease"

COLUMN_RENAMES = {
    "General_Health": "general_health",
    "Checkup": "checkup",
    "Exercise": "exercise",
    "Heart_Disease": "heart_disease",
    "Skin_Cancer": "skin_cancer",
    "Other_Cancer": "other_cancer",
    "Depression": "depression",
    "Diabetes": "diabetes",
    "Arthritis": "arthritis",
    "Sex": "sex",
    "Age_Category": "age_category",
    "Height_(cm)": "height_cm",
    "Weight_(kg)": "weight_kg",
    "BMI": "bmi",
    "Smoking_History": "smoking_history",
    "Alcohol_Consumption": "alcohol_consumption",
    "Fruit_Consumption": "fruit_consumption",
    "Green_Vegetables_Consumption": "green_vegetables_consumption",
    "FriedPotato_Consumption": "fried_potato_consumption",
}

GENERAL_HEALTH_ORDER = ["Poor", "Fair", "Good", "Very Good", "Excellent"]
CHECKUP_ORDER = [
    "Never",
    "5 or more years ago",
    "Within the past 5 years",
    "Within the past 2 years",
    "Within the past year",
]
AGE_ORDER = [
    "18-24",
    "25-29",
    "30-34",
    "35-39",
    "40-44",
    "45-49",
    "50-54",
    "55-59",
    "60-64",
    "65-69",
    "70-74",
    "75-79",
    "80+",
]
DIABETES_RENAMES = {
    "No": "No",
    "No, pre-diabetes or borderline diabetes": "Pre-diabetes",
    "Yes, but female told only during pregnancy": "Gestational only",
    "Yes": "Yes",
}
DIABETES_LEVELS = ["No", "Pre-diabetes", "Gestational only", "Yes"]
SEX_LEVELS = ["Female", "Male"]

ORDERED_LEVELS = {
    "general_health": GENERAL_HEALTH_ORDER,
    "checkup": CHECKUP_ORDER,
    "age_category": AGE_ORDER,
}
BINARY_COLS = [
    "exercise",
    "skin_cancer",
    "other_cancer",
    "depression",
    "arthritis",
    "smoking_history",
]
NUMERIC_COLS = [
    "height_cm",
    "weight_kg",
    "bmi",
    "alcohol_consumption",
    "fruit_consumption",
    "green_vegetables_consumption",
    "fried_potato_consumption",
]
CATEGORICAL_COLS = ["general_health", "checkup", "age_category", "sex", "diabetes", *BINARY_COLS]

# Plausibility limits used to FLAG (never drop) suspicious body measurements.
# Chosen generously: values outside are either data-entry/unit-conversion
# errors or extremely rare. See notebooks/01_eda.ipynb for the reasoning.
PLAUSIBLE_RANGES = {
    "height_cm": (120.0, 220.0),
    "weight_kg": (30.0, 250.0),
    "bmi": (12.0, 70.0),
}

LABELS = {
    "general_health": "Self-rated general health",
    "checkup": "Last routine checkup",
    "exercise": "Exercise in past 30 days",
    "heart_disease": "Heart disease",
    "skin_cancer": "Skin cancer",
    "other_cancer": "Other cancer",
    "depression": "Depression",
    "diabetes": "Diabetes",
    "arthritis": "Arthritis",
    "sex": "Sex",
    "age_category": "Age group",
    "height_cm": "Height (cm)",
    "weight_kg": "Weight (kg)",
    "bmi": "BMI (kg/m²)",
    "smoking_history": "Smoking history",
    "alcohol_consumption": "Alcohol (days in past 30)",
    "fruit_consumption": "Fruit (times per month)",
    "green_vegetables_consumption": "Green vegetables (times per month)",
    "fried_potato_consumption": "Fried potatoes (times per month)",
}


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Read the CSV exactly as distributed on Kaggle."""
    path = Path(path) if path is not None else DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `make data` (or `python scripts/download_data.py`) first."
        )
    return pd.read_csv(path)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a typed, de-duplicated copy of the raw data.

    * exact duplicate rows are dropped (there is no respondent ID, see DECISIONS.md)
    * columns are renamed to snake_case
    * the target becomes an int8 0/1 column
    * ordinal variables become ordered categoricals, the rest plain categoricals
    """
    missing = set(COLUMN_RENAMES) - set(raw.columns)
    if missing:
        raise ValueError(f"Raw data is missing expected columns: {sorted(missing)}")

    df = raw.drop_duplicates().rename(columns=COLUMN_RENAMES).reset_index(drop=True)
    df = df[list(COLUMN_RENAMES.values())]

    df[TARGET] = _yes_no_to_int(df[TARGET], TARGET)
    for col, levels in ORDERED_LEVELS.items():
        df[col] = _to_categorical(df[col], levels, ordered=True, name=col)
    for col in BINARY_COLS:
        df[col] = _to_categorical(df[col], ["No", "Yes"], ordered=False, name=col)
    df["sex"] = _to_categorical(df["sex"], SEX_LEVELS, ordered=False, name="sex")
    df["diabetes"] = _to_categorical(
        df["diabetes"].map(DIABETES_RENAMES), DIABETES_LEVELS, ordered=False, name="diabetes"
    )
    for col in NUMERIC_COLS:
        df[col] = df[col].astype("float64")
    return df


def load_clean(path: str | Path | None = None) -> pd.DataFrame:
    """Load the CSV and apply :func:`clean`."""
    return clean(load_raw(path))


def add_quality_flags(df: pd.DataFrame, bmi_tolerance: float = 0.5) -> pd.DataFrame:
    """Add boolean data-quality flags without removing any rows.

    ``bmi_recomputed``     BMI recomputed from height and weight
    ``flag_bmi_mismatch``  reported BMI differs from the recomputed one by more than the tolerance
    ``flag_implausible``   height, weight or BMI outside ``PLAUSIBLE_RANGES``
    """
    out = df.copy()
    out["bmi_recomputed"] = out["weight_kg"] / (out["height_cm"] / 100.0) ** 2
    out["flag_bmi_mismatch"] = (out["bmi"] - out["bmi_recomputed"]).abs() > bmi_tolerance
    implausible = np.zeros(len(out), dtype=bool)
    for col, (low, high) in PLAUSIBLE_RANGES.items():
        implausible |= ((out[col] < low) | (out[col] > high)).to_numpy()
    out["flag_implausible"] = implausible
    return out


def ordinal_codes(series: pd.Series) -> pd.Series:
    """Integer codes (0..k-1) of an ordered categorical, following its category order."""
    if not (isinstance(series.dtype, pd.CategoricalDtype) and series.cat.ordered):
        raise TypeError(f"{series.name!r} is not an ordered categorical")
    return series.cat.codes.astype("int8").rename(series.name)


def _yes_no_to_int(series: pd.Series, name: str) -> pd.Series:
    mapped = series.map({"No": 0, "Yes": 1})
    if mapped.isna().any():
        bad = sorted(series[mapped.isna()].astype(str).unique())
        raise ValueError(f"Unexpected values in {name!r}: {bad}")
    return mapped.astype("int8")


def _to_categorical(series: pd.Series, levels: list[str], ordered: bool, name: str) -> pd.Series:
    unexpected = set(series.dropna().unique()) - set(levels)
    if unexpected or series.isna().any():
        raise ValueError(f"Unexpected values in {name!r}: {sorted(map(str, unexpected))}")
    return pd.Series(
        pd.Categorical(series, categories=levels, ordered=ordered), index=series.index, name=name
    )
