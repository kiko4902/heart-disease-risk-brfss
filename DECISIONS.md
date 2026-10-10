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
