# Data

The CSV is not versioned. Get it with `make data` (or `python scripts/download_data.py`), which saves `data/CVD_cleaned.csv` and verifies its SHA-256 checksum.

| | |
|---|---|
| Dataset | [Cardiovascular Diseases Risk Prediction Dataset](https://www.kaggle.com/datasets/alphiree/cardiovascular-diseases-risk-prediction-dataset) (Kaggle, user `alphiree`) |
| Origin | CDC Behavioral Risk Factor Surveillance System (BRFSS) 2021, a telephone survey of US adults. The Kaggle author selected 19 of 304 variables and removed incomplete rows. |
| Licence | CC0: Public Domain (as stated on Kaggle) |
| File | `CVD_cleaned.csv`, 32.5 MB, 308,854 rows × 19 columns |
| SHA-256 | `0c3c399c95c9df8ba65fa36821a95c8f27609508dc6927b6a6e7034f25a492c8` |

Manual download: open the Kaggle page, click "Download", unzip, and place `CVD_cleaned.csv` in this folder.

## Column dictionary

`src/cvd/data.py` renames the columns to snake_case; both names are shown.

| Raw column | Cleaned column | Type | Values |
|---|---|---|---|
| `Heart_Disease` | `heart_disease` | target, 0/1 | Respondent reports having been told they have coronary heart disease or a heart attack |
| `General_Health` | `general_health` | ordered | Poor < Fair < Good < Very Good < Excellent (self-rated) |
| `Checkup` | `checkup` | ordered | Time since last routine checkup: Never < 5 or more years ago < Within the past 5 years < Within the past 2 years < Within the past year |
| `Age_Category` | `age_category` | ordered | 13 five-year bands, 18-24 to 80+ |
| `Sex` | `sex` | categorical | Female, Male |
| `Exercise` | `exercise` | Yes/No | Any physical activity outside of work in the past 30 days |
| `Smoking_History` | `smoking_history` | Yes/No | Has a history of smoking |
| `Diabetes` | `diabetes` | categorical | No, Pre-diabetes, Gestational only, Yes (labels shortened from the raw file) |
| `Arthritis`, `Depression`, `Skin_Cancer`, `Other_Cancer` | same, lower case | Yes/No | Ever told they have the condition |
| `Height_(cm)` | `height_cm` | numeric | centimetres |
| `Weight_(kg)` | `weight_kg` | numeric | kilograms |
| `BMI` | `bmi` | numeric | kg/m² |
| `Alcohol_Consumption` | `alcohol_consumption` | numeric, 0–30 | Days with at least one alcoholic drink in the past 30 days |
| `Fruit_Consumption` | `fruit_consumption` | numeric, 0–120 | Times fruit was eaten per month |
| `Green_Vegetables_Consumption` | `green_vegetables_consumption` | numeric, 0–128 | Times green vegetables were eaten per month |
| `FriedPotato_Consumption` | `fried_potato_consumption` | numeric, 0–128 | Times fried potatoes were eaten per month |

## What is verified and what is assumed

The Kaggle page does not document units or exact question wording, so the descriptions above are partly inferred.

- **Verified from the data:** value ranges, category labels, and that `BMI` equals weight / height² (checked in `notebooks/01_eda.ipynb`).
- **Assumed, for the consumption variables:** "times per month". The evidence is the BRFSS 2021 questionnaire, which asks how often fruit, green vegetables and fried potatoes were eaten per day, week or month, and the heaping of values at 30 and 60 (once and twice a day) seen in the EDA notebook.
- **Assumed, for alcohol:** "days in the past 30 with at least one drink", from the BRFSS question and the 0–30 range.
- **Assumed, for the Yes/No variables:** the wording of the corresponding BRFSS questions. In particular it is not documented whether `Heart_Disease` covers coronary heart disease, heart attack, or both, and whether `Smoking_History` means "at least 100 cigarettes in a lifetime" as in BRFSS.

Other things to know: there is no respondent ID, no US state, and no survey weight in this file.
