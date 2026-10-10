"""Synthetic data with the same schema as the Kaggle CSV (the real file is not versioned)."""

import numpy as np
import pandas as pd
import pytest

from cvd.data import AGE_ORDER, CHECKUP_ORDER, DIABETES_RENAMES, GENERAL_HEALTH_ORDER, clean


def make_raw(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    yes_no = ["No", "Yes"]
    height = rng.normal(170, 10, n).round()
    weight = rng.normal(80, 15, n).round(2)
    age_idx = rng.integers(0, len(AGE_ORDER), n)
    # Older respondents get heart disease more often so models have signal to find.
    heart = rng.random(n) < 0.05 + 0.03 * age_idx
    return pd.DataFrame(
        {
            "General_Health": rng.choice(GENERAL_HEALTH_ORDER, n),
            "Checkup": rng.choice(CHECKUP_ORDER, n),
            "Exercise": rng.choice(yes_no, n),
            "Heart_Disease": np.where(heart, "Yes", "No"),
            "Skin_Cancer": rng.choice(yes_no, n),
            "Other_Cancer": rng.choice(yes_no, n),
            "Depression": rng.choice(yes_no, n),
            "Diabetes": rng.choice(list(DIABETES_RENAMES), n),
            "Arthritis": rng.choice(yes_no, n),
            "Sex": rng.choice(["Female", "Male"], n),
            "Age_Category": np.array(AGE_ORDER)[age_idx],
            "Height_(cm)": height,
            "Weight_(kg)": weight,
            "BMI": (weight / (height / 100) ** 2).round(2),
            "Smoking_History": rng.choice(yes_no, n),
            "Alcohol_Consumption": rng.integers(0, 31, n).astype(float),
            "Fruit_Consumption": rng.integers(0, 121, n).astype(float),
            "Green_Vegetables_Consumption": rng.integers(0, 121, n).astype(float),
            "FriedPotato_Consumption": rng.integers(0, 121, n).astype(float),
        }
    )


@pytest.fixture(scope="session")
def raw() -> pd.DataFrame:
    return make_raw()


@pytest.fixture(scope="session")
def df(raw) -> pd.DataFrame:
    return clean(raw)
