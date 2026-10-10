"""One plotting style for every figure in the project."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve

from cvd.data import PROJECT_ROOT

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

BLUE = "#2f6f9f"
RED = "#c8553d"
GREY = "#8a8f98"
LIGHT = "#d9dde3"
MODEL_COLORS = [GREY, BLUE, RED, "#5a9a6f", "#8d6cab"]
TARGET_PALETTE = {0: BLUE, 1: RED, "No": BLUE, "Yes": RED}


def set_style() -> None:
    """Apply the project-wide matplotlib/seaborn style."""
    sns.set_theme(context="notebook", style="whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (8, 4.5),
            "figure.dpi": 100,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": LIGHT,
            "grid.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, name: str) -> Path:
    """Save a figure as PNG under reports/figures/."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    return path


def prevalence_bar(
    table: pd.DataFrame,
    ax: plt.Axes,
    title: str,
    xlabel: str,
    base_rate: float | None = None,
    annotate: bool = True,
    rotate: int = 0,
    ylabel: bool = True,
) -> plt.Axes:
    """Bar chart of target prevalence per level (expects a ``prevalence`` column)."""
    labels = [str(i) for i in table.index]
    bars = ax.bar(labels, table["prevalence"], color=BLUE, width=0.75)
    if base_rate is not None:
        ax.axhline(
            base_rate, color=RED, linestyle="--", linewidth=1.2, label=f"Overall {base_rate:.1%}"
        )
        ax.legend(loc="best", fontsize=8, handlelength=1.6)
    if annotate:
        for bar, value in zip(bars, table["prevalence"], strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.1%}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Heart disease prevalence" if ylabel else "")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    if rotate:
        ax.tick_params(axis="x", rotation=rotate)
        for label in ax.get_xticklabels():
            label.set_ha("right")
    return ax


def forest_plot(
    table: pd.DataFrame,
    ax: plt.Axes,
    title: str,
    estimate: str = "odds_ratio",
    xlabel: str = "Odds ratio (log scale, 95% CI)",
    color: str = BLUE,
    label: str | None = None,
    offset: float = 0.0,
) -> plt.Axes:
    """Forest plot of ratio estimates with CIs (columns: estimate, ci_low, ci_high)."""
    y = np.arange(len(table))[::-1] + offset
    est = table[estimate].to_numpy()
    err = np.vstack([est - table["ci_low"].to_numpy(), table["ci_high"].to_numpy() - est])
    ax.errorbar(est, y, xerr=err, fmt="o", color=color, markersize=4, linewidth=1.3, label=label)
    ax.axvline(1.0, color=GREY, linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yticks(np.arange(len(table))[::-1])
    ax.set_yticklabels(table.index)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    return ax


def pr_curves(y_true, probabilities: dict[str, np.ndarray], ax: plt.Axes, title: str) -> plt.Axes:
    """Precision-recall curves for several models with the base rate as reference."""
    base_rate = float(np.mean(y_true))
    for (name, prob), color in zip(probabilities.items(), MODEL_COLORS[1:], strict=False):
        precision, recall, _ = precision_recall_curve(y_true, prob)
        ap = average_precision_score(y_true, prob)
        ax.plot(recall, precision, color=color, linewidth=1.8, label=f"{name} (PR-AUC {ap:.3f})")
    ax.axhline(
        base_rate, color=GREY, linestyle="--", linewidth=1, label=f"Base rate {base_rate:.3f}"
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    return ax


def calibration_plot(
    y_true, probabilities: dict[str, np.ndarray], ax: plt.Axes, title: str, n_bins: int = 10
) -> plt.Axes:
    """Reliability diagram using quantile bins."""
    upper = 0.0
    for (name, prob), color in zip(probabilities.items(), MODEL_COLORS[1:], strict=False):
        observed, predicted = calibration_curve(y_true, prob, n_bins=n_bins, strategy="quantile")
        ax.plot(predicted, observed, marker="o", markersize=4, color=color, label=name)
        upper = max(upper, predicted.max(), observed.max())
    upper = min(1.0, upper * 1.08)
    ax.plot([0, upper], [0, upper], color=GREY, linestyle="--", linewidth=1, label="Perfect")
    ax.set_xlabel("Mean predicted probability (decile bins)")
    ax.set_ylabel("Observed prevalence")
    ax.set_title(title)
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.legend(loc="upper left")
    return ax


def horizontal_bar(
    values: pd.Series, ax: plt.Axes, title: str, xlabel: str, color: str = BLUE
) -> plt.Axes:
    """Sorted horizontal bar chart (largest on top)."""
    ordered = values.sort_values()
    ax.barh([str(i) for i in ordered.index], ordered.to_numpy(), color=color)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    return ax


def prevalence_grid(
    df: pd.DataFrame,
    columns: list[str],
    target: str,
    labels: dict[str, str],
    ncols: int = 2,
    suptitle: str | None = None,
) -> plt.Figure:
    """Grid of prevalence bar charts, one panel per categorical column, shared y-axis."""
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.6 * nrows), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    base_rate = df[target].mean()
    for i, (ax, col) in enumerate(zip(axes, columns, strict=False)):
        table = df.groupby(col, observed=True)[target].agg(prevalence="mean")
        rotate = 20 if max(len(str(i)) for i in table.index) > 12 else 0
        prevalence_bar(
            table, ax, labels[col], "", base_rate=base_rate, rotate=rotate, ylabel=i % ncols == 0
        )
    for ax in axes[len(columns) :]:
        ax.set_visible(False)
    if suptitle:
        fig.suptitle(suptitle, x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def discrete_distribution(
    series: pd.Series, ax: plt.Axes, title: str, xlabel: str, max_value: float | None = None
) -> plt.Axes:
    """Share of respondents at each distinct value (shows heaping in count-like variables)."""
    shares = series.value_counts(normalize=True).sort_index()
    if max_value is not None:
        shares = shares[shares.index <= max_value]
    ax.bar(shares.index, shares.to_numpy(), color=BLUE, width=0.9)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Share of respondents")
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    return ax


def histogram_by_target(
    df: pd.DataFrame, column: str, target: str, ax: plt.Axes, title: str, xlabel: str, bins
) -> plt.Axes:
    """Overlaid density histograms of a numeric column for target = 0 and 1."""
    for value, name in [(0, "No heart disease"), (1, "Heart disease")]:
        ax.hist(
            df.loc[df[target] == value, column],
            bins=bins,
            density=True,
            alpha=0.55,
            color=TARGET_PALETTE[value],
            label=name,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    return ax
