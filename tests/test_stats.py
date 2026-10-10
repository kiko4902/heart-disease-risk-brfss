import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests

from cvd import stats


def test_cramers_v_perfect_and_no_association():
    x = pd.Series(["a", "b"] * 50)
    assert stats.cramers_v(x, x, bias_correction=False) == pytest.approx(1.0)
    y = pd.Series(["u", "u", "v", "v"] * 25)
    assert stats.cramers_v(x, y) == pytest.approx(0.0)


def test_cramers_v_known_2x2():
    # 2x2 table [[30, 10], [10, 30]]: chi2 = 20, n = 80, V = sqrt(20/80) = 0.5
    x = pd.Series(["a"] * 40 + ["b"] * 40)
    y = pd.Series(["u"] * 30 + ["v"] * 10 + ["u"] * 10 + ["v"] * 30)
    assert stats.cramers_v(x, y, bias_correction=False) == pytest.approx(0.5)
    result = stats.chi2_test(x, y)
    assert result["chi2"] == pytest.approx(20.0)
    assert result["dof"] == 1
    assert result["min_expected"] == pytest.approx(20.0)


def test_risk_ratio_and_difference_known_values():
    rr = stats.risk_ratio(20, 100, 10, 100)
    assert rr["risk_ratio"] == pytest.approx(2.0)
    se = np.sqrt(1 / 20 - 1 / 100 + 1 / 10 - 1 / 100)
    assert rr["ci_low"] == pytest.approx(np.exp(np.log(2) - 1.959964 * se), rel=1e-5)
    assert rr["ci_low"] < 2.0 < rr["ci_high"]
    rd = stats.risk_difference(20, 100, 10, 100)
    assert rd["risk_difference"] == pytest.approx(0.10)
    assert rd["ci_low"] < 0.10 < rd["ci_high"]


def test_risk_ratio_requires_events():
    with pytest.raises(ValueError):
        stats.risk_ratio(0, 100, 10, 100)


def test_prevalence_table():
    df = pd.DataFrame(
        {"g": ["a"] * 100 + ["b"] * 100, "y": [1] * 10 + [0] * 90 + [1] * 30 + [0] * 70}
    )
    table = stats.prevalence_table(df, "g", "y")
    assert table.loc["a", "prevalence"] == pytest.approx(0.10)
    assert table.loc["a", "rr"] == 1.0
    assert table.loc["b", "rr"] == pytest.approx(3.0)
    assert table.loc["b", "rd"] == pytest.approx(0.20)
    assert stats.prevalence_table(df, "g", "y", reference="b").loc["a", "rr"] == pytest.approx(
        1 / 3
    )


def test_cohens_d_known_value():
    a = np.array([2.0, 4.0, 6.0])
    b = np.array([1.0, 3.0, 5.0])
    assert stats.cohens_d(a, b) == pytest.approx(0.5)  # mean diff 1, pooled sd 2
    assert stats.cohens_d(b, a) == pytest.approx(-0.5)


def test_rank_biserial_extremes_and_agreement_with_scipy():
    assert stats.rank_biserial([4, 5, 6], [1, 2, 3]) == pytest.approx(1.0)
    assert stats.rank_biserial([1, 2, 3], [4, 5, 6]) == pytest.approx(-1.0)
    assert stats.rank_biserial([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 10, 200), rng.integers(0, 12, 300)  # with ties
    u = sps.mannwhitneyu(a, b).statistic
    assert stats.rank_biserial(a, b) == pytest.approx(2 * u / (200 * 300) - 1)


def test_bootstrap_ci_is_seeded_and_covers_estimate():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0.5, 1, 400), rng.normal(0, 1, 400)
    first = stats.bootstrap_ci(a, b, stats.cohens_d, n_boot=200)
    second = stats.bootstrap_ci(a, b, stats.cohens_d, n_boot=200)
    assert first == second
    assert first[0] < stats.cohens_d(a, b) < first[1]
    assert first != stats.bootstrap_ci(a, b, stats.cohens_d, n_boot=200, seed=7)


def test_holm_matches_statsmodels():
    p = [0.01, 0.04, 0.03, 0.005, 0.5]
    assert stats.holm_adjust(p) == pytest.approx(multipletests(p, method="holm")[1])
    assert stats.holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_partial_eta_squared():
    table = pd.DataFrame({"sum_sq": [10.0, 30.0, 60.0]}, index=["A", "B", "Residual"])
    eta = stats.partial_eta_squared(table)
    assert eta["A"] == pytest.approx(10 / 70)
    assert eta["B"] == pytest.approx(30 / 90)
    assert "Residual" not in eta.index


def test_format_p_never_prints_zero():
    assert stats.format_p(0.0) == "< 1e-300"
    assert stats.format_p(1e-12) == "1.0e-12"
    assert stats.format_p(0.0312) == "0.031"
