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


def binned_prevalence(
    df: pd.DataFrame, column: str, target: str, bins: list[float], labels: list[str]
) -> pd.DataFrame:
    """Prevalence of a 0/1 target within bins of a numeric column (left-closed bins)."""
    binned = pd.cut(df[column], bins=bins, labels=labels, right=False)
    table = df.groupby(binned, observed=True)[target].agg(n="size", prevalence="mean")
    table.index.name = column
    return table


def wilson_ci(events: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    z = stats.norm.ppf(1 - alpha / 2)
    p = events / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return float(centre - half), float(centre + half)


def format_p(p: float) -> str:
    """Human-readable p-value that never prints a misleading 0.0."""
    if p < 1e-300:
        return "< 1e-300"
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.3f}"


def categorical_association_table(
    df: pd.DataFrame, contrasts: dict[str, tuple[str, str]], target: str
) -> pd.DataFrame:
    """Chi-square test, Cramér's V and one headline contrast per categorical factor.

    ``contrasts`` maps each factor to ``(exposed_level, reference_level)``. The risk
    ratio and risk difference (95% CI) compare those two levels.
    """
    rows = []
    for factor, (exposed, reference) in contrasts.items():
        test = chi2_test(df[factor], df[target])
        levels = prevalence_table(df, factor, target, reference=reference)
        rows.append(
            {
                "factor": factor,
                "levels": len(levels),
                "chi2": test["chi2"],
                "dof": test["dof"],
                "p_value": test["p_value"],
                "cramers_v": test["cramers_v"],
                "min_expected": test["min_expected"],
                "contrast": f"{exposed} vs {reference}",
                "prev_exposed": levels.loc[exposed, "prevalence"],
                "prev_reference": levels.loc[reference, "prevalence"],
                "risk_ratio": levels.loc[exposed, "rr"],
                "rr_low": levels.loc[exposed, "rr_low"],
                "rr_high": levels.loc[exposed, "rr_high"],
                "risk_diff": levels.loc[exposed, "rd"],
                "rd_low": levels.loc[exposed, "rd_low"],
                "rd_high": levels.loc[exposed, "rd_high"],
            }
        )
    return pd.DataFrame(rows).set_index("factor")


def numeric_association_table(
    df: pd.DataFrame,
    columns: list[str],
    target: str,
    n_boot: int = 1000,
    boot_sample: int | None = 50_000,
    seed: int = SEED,
) -> pd.DataFrame:
    """Compare each numeric column between target = 1 and target = 0.

    Point estimates (Mann-Whitney U p-value, rank-biserial r, Cohen's d) use all
    rows. The percentile bootstrap CIs use a seeded random subsample of
    ``boot_sample`` rows to keep the run time reasonable; pass ``None`` to
    bootstrap the full data.
    """
    boot_df = (
        df
        if boot_sample is None or boot_sample >= len(df)
        else df.sample(boot_sample, random_state=seed)
    )
    rows = []
    for col in columns:
        pos, neg = df.loc[df[target] == 1, col], df.loc[df[target] == 0, col]
        boot_pos = boot_df.loc[boot_df[target] == 1, col]
        boot_neg = boot_df.loc[boot_df[target] == 0, col]
        r_low, r_high = bootstrap_ci(boot_pos, boot_neg, rank_biserial, n_boot=n_boot, seed=seed)
        d_low, d_high = bootstrap_ci(boot_pos, boot_neg, cohens_d, n_boot=n_boot, seed=seed)
        rows.append(
            {
                "variable": col,
                "mean_hd": pos.mean(),
                "mean_no_hd": neg.mean(),
                "median_hd": pos.median(),
                "median_no_hd": neg.median(),
                "p_value": stats.mannwhitneyu(pos, neg, alternative="two-sided").pvalue,
                "rank_biserial": rank_biserial(pos, neg),
                "rb_low": r_low,
                "rb_high": r_high,
                "cohens_d": cohens_d(pos, neg),
                "d_low": d_low,
                "d_high": d_high,
            }
        )
    return pd.DataFrame(rows).set_index("variable")


def mean_difference_ci(a, b, alpha: float = 0.05) -> dict[str, float]:
    """Difference in means (a - b) with a Welch confidence interval and p-value."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    result = stats.ttest_ind(a, b, equal_var=False)
    ci = result.confidence_interval(1 - alpha)
    return {
        "mean_diff": float(a.mean() - b.mean()),
        "ci_low": float(ci.low),
        "ci_high": float(ci.high),
        "p_value": float(result.pvalue),
        "cohens_d": cohens_d(a, b),
    }
