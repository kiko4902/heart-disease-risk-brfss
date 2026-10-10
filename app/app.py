"""Streamlit app: estimated probability of reported heart disease with a SHAP explanation.

Run with `make app`. The trained models are versioned in app/model/ and are
recreated by `make notebooks`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

try:
    import cvd  # noqa: F401
except ImportError:  # not installed as a package, e.g. on Streamlit Community Cloud
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvd import models, plots  # noqa: E402
from cvd.data import (  # noqa: E402
    AGE_ORDER,
    CHECKUP_ORDER,
    DIABETES_LEVELS,
    GENERAL_HEALTH_ORDER,
    LABELS,
)

MODEL_LABELS = {
    "all": "All predictors (lifestyle, demographics and health status)",
    "lifestyle": "Lifestyle and demographics only",
}
DISCLAIMER = (
    "**Educational project, not medical advice.** This model was trained on a 2021 US telephone "
    "survey and estimates how often people who gave similar answers *reported* existing heart "
    "disease. It cannot diagnose anything or predict your future health. If you have concerns "
    "about your heart, talk to a doctor."
)


@st.cache_resource
def load_bundle() -> dict:
    return models.load_bundle()


def build_row(answers: dict) -> pd.DataFrame:
    """Turn the form answers into a one-row data frame with the model's column names."""
    row = dict(answers)
    row["bmi"] = row["weight_kg"] / (row["height_cm"] / 100) ** 2
    return pd.DataFrame([row])


def predict(bundle: dict, model_key: str, row: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Predicted probability and per-feature SHAP contributions (log-odds), largest first."""
    pipeline = bundle["models"][model_key]
    features = bundle["features"][model_key]
    probability = float(pipeline.predict_proba(row[features])[0, 1])
    values, _ = models.tree_contributions(pipeline, row[features])
    contributions = pd.DataFrame(
        {
            "factor": [LABELS[f] for f in features],
            "your answer": models.display_values(row[features])[0],
            "contribution": values.iloc[0].to_numpy(),
        }
    )
    order = contributions["contribution"].abs().sort_values(ascending=False).index
    return probability, contributions.loc[order].reset_index(drop=True)


def contribution_chart(contributions: pd.DataFrame, top: int = 8) -> plt.Figure:
    plots.set_style()
    shown = contributions.head(top).iloc[::-1]
    labels = [f"{f}: {v}" for f, v in zip(shown["factor"], shown["your answer"], strict=True)]
    colors = [plots.RED if c > 0 else plots.BLUE for c in shown["contribution"]]
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(shown) + 1.2))
    ax.barh(labels, shown["contribution"], color=colors)
    ax.axvline(0, color=plots.GREY, linewidth=1)
    ax.set_xlabel("Contribution to the estimate (log-odds)\n← lowers it        raises it →")
    ax.set_title("What moved this estimate most")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def input_form(bundle: dict, model_key: str) -> dict | None:
    """Render the form for the chosen model; returns the answers once submitted."""
    features = bundle["features"][model_key]
    with st.form("inputs"):
        st.subheader("About you")
        left, right = st.columns(2)
        answers = {
            "sex": left.radio(LABELS["sex"], ["Female", "Male"], horizontal=True),
            "age_category": right.selectbox(LABELS["age_category"], AGE_ORDER, index=7),
            "height_cm": left.number_input(LABELS["height_cm"], 120.0, 220.0, 170.0, step=1.0),
            "weight_kg": right.number_input(LABELS["weight_kg"], 30.0, 250.0, 80.0, step=1.0),
        }

        st.subheader("Lifestyle")
        left, right = st.columns(2)
        answers["smoking_history"] = left.radio(
            "Have you ever smoked regularly?", ["No", "Yes"], horizontal=True
        )
        answers["exercise"] = right.radio(
            "Any exercise outside work in the past 30 days?",
            ["No", "Yes"],
            index=1,
            horizontal=True,
        )
        answers["alcohol_consumption"] = float(
            left.slider("Days with at least one alcoholic drink, past 30 days", 0, 30, 2)
        )
        answers["fruit_consumption"] = float(right.slider("Fruit, times per month", 0, 120, 30))
        answers["green_vegetables_consumption"] = float(
            left.slider("Green vegetables, times per month", 0, 120, 12)
        )
        answers["fried_potato_consumption"] = float(
            right.slider("Fried potatoes, times per month", 0, 120, 4)
        )

        if "general_health" in features:
            st.subheader("Health status")
            left, right = st.columns(2)
            answers["general_health"] = left.select_slider(
                LABELS["general_health"], GENERAL_HEALTH_ORDER, value="Good"
            )
            answers["checkup"] = right.selectbox(
                LABELS["checkup"], CHECKUP_ORDER, index=len(CHECKUP_ORDER) - 1
            )
            answers["diabetes"] = left.selectbox(LABELS["diabetes"], DIABETES_LEVELS)
            for column, condition in zip(
                st.columns(4),
                ["arthritis", "depression", "skin_cancer", "other_cancer"],
                strict=True,
            ):
                answers[condition] = "Yes" if column.checkbox(LABELS[condition]) else "No"

        submitted = st.form_submit_button("Estimate", type="primary")
    return answers if submitted else None


def show_result(bundle: dict, model_key: str, answers: dict) -> None:
    row = build_row(answers)
    probability, contributions = predict(bundle, model_key, row)
    base_rate = bundle["base_rate"]

    st.subheader("Result")
    left, middle, right = st.columns(3)
    left.metric("Estimated probability", f"{probability:.1%}")
    middle.metric("Average in the survey", f"{base_rate:.1%}")
    right.metric("Relative to average", f"{probability / base_rate:.1f}×")
    st.caption(
        f"BMI computed from height and weight: {row['bmi'].iloc[0]:.1f} kg/m². "
        f"Among survey respondents the model scored at about {probability:.0%}, roughly that share "
        "reported heart disease (the model is well calibrated on held-out data)."
    )

    st.pyplot(contribution_chart(contributions))
    st.caption(
        "Bars are SHAP values: how much each answer moved the model's estimate away from the "
        "survey average. They describe the model, not cause and effect. Changing an answer here "
        "does not show what would happen to a person's health."
    )
    with st.expander("All contributions"):
        st.dataframe(
            contributions.assign(contribution=contributions["contribution"].round(3)),
            hide_index=True,
            width="stretch",
        )


def main() -> None:
    st.set_page_config(page_title="Heart disease risk (BRFSS 2021)", page_icon="🫀")
    st.title("Heart disease risk estimate")
    st.warning(DISCLAIMER, icon="⚠️")

    try:
        bundle = load_bundle()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    with st.sidebar:
        st.header("Model")
        model_key = st.radio(
            "Which predictors to use", list(MODEL_LABELS), format_func=MODEL_LABELS.get
        )
        metrics = bundle["test_metrics"][model_key]
        st.markdown(
            f"**Held-out performance**\n\n"
            f"- PR-AUC: {metrics['pr_auc']:.3f} (base rate {bundle['base_rate']:.3f})\n"
            f"- ROC-AUC: {metrics['roc_auc']:.3f}\n"
            f"- Brier score: {metrics['brier']:.4f}\n\n"
            f"LightGBM trained on {bundle['n_train']:,} BRFSS 2021 respondents."
        )
        st.markdown(
            "The *lifestyle only* model leaves out self-rated health, checkups and diagnoses, "
            "which people often know only after becoming a patient."
        )

    answers = input_form(bundle, model_key)
    if answers is not None:
        show_result(bundle, model_key, answers)
    else:
        st.info("Fill in the form and press **Estimate**.")


if __name__ == "__main__":
    main()
