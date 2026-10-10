import pandas as pd
import pytest

from cvd import data


def test_clean_drops_exact_duplicates(raw):
    doubled = pd.concat([raw, raw.iloc[:25]], ignore_index=True)
    assert len(data.clean(doubled)) == len(data.clean(raw)) == len(raw.drop_duplicates())


def test_clean_renames_and_types(df):
    assert list(df.columns) == list(data.COLUMN_RENAMES.values())
    assert df[data.TARGET].dtype == "int8"
    assert set(df[data.TARGET].unique()) <= {0, 1}
    assert not df.isna().any().any()
    for col in data.NUMERIC_COLS:
        assert df[col].dtype == "float64"


def test_ordered_categoricals_have_the_documented_order(df):
    for col, levels in data.ORDERED_LEVELS.items():
        assert df[col].cat.ordered
        assert list(df[col].cat.categories) == levels
    assert df["general_health"].min() == "Poor"
    assert df["age_category"].max() == "80+"
    assert not df["sex"].cat.ordered


def test_target_matches_raw_labels(raw):
    cleaned = data.clean(raw)
    assert cleaned[data.TARGET].sum() == (raw.drop_duplicates()["Heart_Disease"] == "Yes").sum()


def test_diabetes_labels_are_shortened(df):
    assert set(df["diabetes"].cat.categories) == set(data.DIABETES_LEVELS)


def test_clean_rejects_unknown_category(raw):
    broken = raw.copy()
    broken.loc[0, "General_Health"] = "Superb"
    with pytest.raises(ValueError, match="general_health"):
        data.clean(broken)


def test_clean_rejects_missing_column(raw):
    with pytest.raises(ValueError, match="missing expected columns"):
        data.clean(raw.drop(columns="BMI"))


def test_quality_flags_flag_but_never_drop(df):
    modified = df.copy()
    modified.loc[0, "height_cm"] = 95.0  # implausibly short adult
    modified.loc[1, "bmi"] = modified.loc[1, "bmi"] + 10  # inconsistent with height/weight
    flagged = data.add_quality_flags(modified)
    assert len(flagged) == len(modified)
    assert flagged.loc[0, "flag_implausible"]
    assert flagged.loc[1, "flag_bmi_mismatch"]
    assert not flagged.loc[2:, "flag_bmi_mismatch"].any()


def test_ordinal_codes(df):
    codes = data.ordinal_codes(df["general_health"])
    assert codes[df["general_health"] == "Poor"].eq(0).all()
    assert codes[df["general_health"] == "Excellent"].eq(4).all()
    with pytest.raises(TypeError):
        data.ordinal_codes(df["sex"])


def test_load_raw_missing_file_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="make data"):
        data.load_raw(tmp_path / "nope.csv")
