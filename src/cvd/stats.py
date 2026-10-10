"""Effect sizes, confidence intervals and multiple-testing helpers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats

from cvd import SEED


def cramers_v(x: pd.Series, y: pd.Series, bias_correction: bool = True) -> float:
    """Cramér's V for two categorical variables (Bergsma bias correction by default)."""
    table = pd.crosstab(x, y).to_numpy()
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.sum()
    r, k = table.shape
    phi2 = chi2 / n
    if not bias_correction:
        return float(np.sqrt(phi2 / (min(r, k) - 1)))
    phi2_corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(r_corr, k_corr) - 1
    return float(np.sqrt(phi2_corr / denom)) if denom > 0 else 0.0


def chi2_test(x: pd.Series, y: pd.Series) -> dict[str, float]:
    """Pearson chi-square test of independence with Cramér's V."""
    table = pd.crosstab(x, y).to_numpy()
    chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
    return {
        "chi2": float(chi2),
        "dof": int(dof),
        "p_value": float(p),
        "cramers_v": cramers_v(x, y),
        "min_expected": float(expected.min()),
    }


def risk_ratio(
    events_exposed: int, n_exposed: int, events_ref: int, n_ref: int, alpha: float = 0.05
) -> dict[str, float]:
    """Risk ratio (exposed / reference) with a log-scale Wald confidence interval."""
    if min(events_exposed, events_ref) <= 0:
        raise ValueError("Risk ratio needs at least one event in each group")
    p1, p0 = events_exposed / n_exposed, events_ref / n_ref
    rr = p1 / p0
    se = np.sqrt(1 / events_exposed - 1 / n_exposed + 1 / events_ref - 1 / n_ref)
    z = stats.norm.ppf(1 - alpha / 2)
    return {
        "risk_ratio": float(rr),
        "ci_low": float(np.exp(np.log(rr) - z * se)),
        "ci_high": float(np.exp(np.log(rr) + z * se)),
    }


def risk_difference(
    events_exposed: int, n_exposed: int, events_ref: int, n_ref: int, alpha: float = 0.05
) -> dict[str, float]:
    """Difference in proportions (exposed - reference) with a Wald confidence interval."""
    p1, p0 = events_exposed / n_exposed, events_ref / n_ref
    se = np.sqrt(p1 * (1 - p1) / n_exposed + p0 * (1 - p0) / n_ref)
    z = stats.norm.ppf(1 - alpha / 2)
    diff = p1 - p0
    return {
        "risk_difference": float(diff),
        "ci_low": float(diff - z * se),
        "ci_high": float(diff + z * se),
    }


def prevalence_table(df: pd.DataFrame, factor: str, target: str, reference=None) -> pd.DataFrame:
    """Prevalence of a 0/1 target per level of ``factor``.

    Adds the risk ratio and risk difference (95% CI) of each level against the
    reference level (the first level unless ``reference`` is given).
    """
    grouped = df.groupby(factor, observed=True)[target].agg(n="size", events="sum")
    grouped["prevalence"] = grouped["events"] / grouped["n"]
    reference = grouped.index[0] if reference is None else reference
    ref = grouped.loc[reference]
    rows = []
    for level, row in grouped.iterrows():
        if level == reference:
            rows.append(
                {"rr": 1.0, "rr_low": np.nan, "rr_high": np.nan}
                | {"rd": 0.0, "rd_low": np.nan, "rd_high": np.nan}
            )
            continue
        rr = risk_ratio(row["events"], row["n"], ref["events"], ref["n"])
        rd = risk_difference(row["events"], row["n"], ref["events"], ref["n"])
        rows.append(
            {
                "rr": rr["risk_ratio"],
                "rr_low": rr["ci_low"],
                "rr_high": rr["ci_high"],
                "rd": rd["risk_difference"],
                "rd_low": rd["ci_low"],
                "rd_high": rd["ci_high"],
            }
        )
    out = pd.concat([grouped, pd.DataFrame(rows, index=grouped.index)], axis=1)
    out.index.name = factor
    return out


def cohens_d(a, b) -> float:
    """Cohen's d (a - b) using the pooled standard deviation."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled)


def rank_biserial(a, b) -> float:
    """Rank-biserial correlation for two independent samples.

    Equals P(a > b) - P(a < b); ranges from -1 to 1, 0 means no stochastic difference.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ranks = stats.rankdata(np.concatenate([a, b]))
    u = ranks[: len(a)].sum() - len(a) * (len(a) + 1) / 2
    return float(2 * u / (len(a) * len(b)) - 1)


def bootstrap_ci(
    a,
    b,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = SEED,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a two-sample statistic, resampling within each group."""
    a, b = np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot)
    for i in range(n_boot):
        estimates[i] = statistic(
            a[rng.integers(0, len(a), len(a))], b[rng.integers(0, len(b), len(b))]
        )
    low, high = np.quantile(estimates, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def holm_adjust(p_values) -> np.ndarray:
    """Holm step-down adjusted p-values (controls the family-wise error rate)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def partial_eta_squared(anova_table: pd.DataFrame) -> pd.Series:
    """Partial eta squared for each term of a statsmodels ANOVA table."""
    ss_resid = anova_table.loc["Residual", "sum_sq"]
    effects = anova_table.drop(index="Residual")
    return (effects["sum_sq"] / (effects["sum_sq"] + ss_resid)).rename("partial_eta_sq")


def format_p(p: float) -> str:
    """Human-readable p-value that never prints a misleading 0.0."""
    if p < 1e-300:
        return "< 1e-300"
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.3f}"
