# Decisions

Judgment calls made while building the project, one line each: what was decided and why.

## Setup and data

- **Dropped the 80 exact duplicate rows.** There is no respondent ID, so true duplicates cannot be told apart from two people with identical answers; 80 of 308,854 rows (0.03%) cannot change any result, and dropping matches the original exam notebook.
- **Renamed columns to snake_case** (`Height_(cm)` → `height_cm`). Parentheses break model formulas and the original names are inconsistent (`FriedPotato_Consumption`).
- **Target stored as int 0/1; other Yes/No variables stay categorical.** Prevalence is then a plain mean, and the predictors keep readable labels in tables and plots.
- **Shortened the diabetes labels** ("Yes, but female told only during pregnancy" → "Gestational only"). Needed for readable axes; the mapping is in `src/cvd/data.py`.
- **`requirements.in` plus a compiled `requirements.txt`.** The pinned file is a universal `uv pip compile` lock, so the same file installs on Windows, macOS and Linux (Docker). `requirements.in` lists the direct dependencies.
- **`make setup` uses `uv` when available and falls back to `python3.12 -m venv` + pip.** `uv` can fetch Python 3.12 itself; the fallback keeps the Makefile usable without it.
- **Download script uses Kaggle's public download URL, not the `kaggle` package.** The dataset is CC0 and downloads without credentials; credentials are sent if present, and manual instructions are printed on failure. The file is verified by SHA-256.
- **Added `archive/REVIEW.md`** next to the original notebook. The README links to it to show what was wrong with the exam version and why the project was rebuilt.
- **Consumption units are an assumption** ("times per month"; alcohol "days in past 30"). The Kaggle page does not document them; see `data/README.md` for the evidence.

## Exploration and statistics

- **Outliers are flagged, not dropped** (height outside 120–220 cm, weight outside 30–250 kg, BMI outside 12–70). Only 234 rows are affected, tree models are insensitive to them, and notebook 3 shows that removing them changes no odds ratio by more than about 1%.
- **Height and weight are excluded from all models; BMI is kept.** BMI is computed from them, and height is mostly a proxy for sex (the confounding sidebar).
- **Mann-Whitney U is the primary test for numeric predictors**, with rank-biserial r as its effect size and Cohen's d alongside. The consumption variables are discrete, skewed and heaped, so a rank-based comparison is the better fit.
- **Bias-corrected Cramér's V** (Bergsma). The correction is negligible at this sample size, but it is the better default and keeps the helper correct for small tables.
- **One pre-chosen contrast per categorical variable** for risk ratios (ends of the scale for ordinal variables). The checkup contrast uses "5 or more years ago", not "Never", because "Never" has only 1,407 respondents.
- **Bootstrap CIs use a seeded 50,000-row subsample with 1,000 resamples.** A full-data bootstrap of a rank statistic takes minutes for no practical gain; the subsample makes the intervals wider, which is the conservative direction. Point estimates use all rows.
- **The Holm family is the 16 predictor-versus-target tests.** The confounding sidebar is a teaching example, not part of the confirmatory analysis, so its tests are not in the family.

## Logistic regression

- **Reference levels:** age 18-24, Female, "Excellent" health, checkup "Within the past year", "No" for yes/no variables. These are the natural or largest groups, which keeps the intervals narrow.
- **Numeric predictors are rescaled** (BMI per 5 kg/m², alcohol per 10 days, food per one extra time a day) so each odds ratio describes a meaningful step.
- **Two feature sets.** "Lifestyle" is sex, age, BMI, smoking, exercise, alcohol and diet; "all" adds general health, checkup and the diagnoses. Alcohol, exercise and BMI stay in the lifestyle set even though a diagnosis can change them too, because they are what a person could report without any medical history; the notebooks say so explicitly.
- **BMI and alcohol stay linear in the main models** although the check in notebook 3 shows both are non-linear. Linear terms keep the forest plot readable; the caveat and the categorical refit are in the notebook.

## Predictive model

- **80/20 stratified split, 5-fold stratified CV on the training part, seed 42.** The test set is scored once.
- **LightGBM, not XGBoost.** Faster on this data on a laptop and exact fast SHAP values; no accuracy difference is expected between the two here.
- **No class weights and no resampling; the imbalance is handled by the decision threshold.** In cross-validation class weights left PR-AUC unchanged and nearly tripled the Brier score, and the app needs calibrated probabilities.
- **Threshold maximises F2 on out-of-fold training predictions.** F2 encodes the assumption that a missed case costs more than a false alarm; alternatives are tabulated in the notebook.
- **Only tree size was tuned (7, 15, 31 leaves); 7 was selected.** Logistic regression already matches boosting to within noise, so a larger search could not change any conclusion.
- **LightGBM is the saved model even though logistic regression performs the same.** It handles the non-linear BMI and alcohol effects without manual coding and gives exact per-person SHAP explanations for the app.
- **Tree features use one integer column per variable** (ordered codes; diabetes ordered by severity) instead of one-hot columns, so SHAP reports one contribution per original variable.
- **SHAP values are computed on a seeded sample of 5,000 test rows.** Enough for stable global importance, and it keeps the notebook fast.
- **PR-AUC intervals are a 500-resample bootstrap of the test rows.**

## App

- **The app offers both saved models, defaulting to "all predictors".** The lifestyle-only option lets a user see an estimate that does not depend on self-rated health or diagnoses, which mirrors the comparison in notebooks 3 and 4.
- **The form asks for height and weight and computes BMI.** People know their height and weight, not their BMI; inputs are limited to the plausibility ranges from notebook 1.
- **SHAP contributions are shown in log-odds, as a bar chart of the eight largest.** Log-odds contributions add up exactly; converting each to percentage points would not.
- **The Docker image copies `models/` from the build context and installs the full `requirements.txt`.** One lock file for everything keeps versions identical between training and serving, at the cost of a larger image; the data is not needed at run time.
- **App tests train a tiny model on synthetic data** (Streamlit `AppTest`), so the test suite runs without the dataset or the saved models.
