# Model card: crop recommendation classifier

## Overview
- **Task:** multi-class classification. Given seven soil and climate measurements, rank the 22 crops by suitability.
- **Selected model:** Random Forest (chosen by 5-fold cross-validation on the training split).
- **Intended use:** education and demonstration of an ML decision-support workflow.
- **Out of scope:** yield forecasts, fertilizer prescriptions, or any decision about real land without local expert advice.

## Data
- 2200 rows, 22 crops, 100 examples per crop.
- SHA-256 of the CSV used: `54a5a6e5408668e668667efc50de2fc867c1b875e0431b4f54dd331b0a109a4e`

| Input | Range in training data |
|---|---|
| Nitrogen (N) | 0 to 140 |
| Phosphorus (P) | 5 to 145 |
| Potassium (K) | 5 to 205 |
| Temperature | 8.82567 to 43.6755 |
| Humidity | 14.258 to 99.9819 |
| Soil pH | 3.50475 to 9.93509 |
| Rainfall | 20.2113 to 298.56 |

## Training and evaluation
- Stratified split: 1320 train / 440 calibration / 440 test, seed 42. The calibration split is used only for the conformal prediction sets below.
- The test set is used once, to report the selected model. It is never used to choose between models.
- **Held-out test accuracy: 0.998. Macro F1: 0.998.**

| Model | CV accuracy (mean ± std) | Test accuracy | Test macro F1 |
|---|---|---|---|
| Random Forest **(selected)** | 0.997 ± 0.002 | 0.998 | 0.998 |
| Gaussian Naive Bayes | 0.996 ± 0.002 | 0.995 | 0.995 |
| Decision Tree | 0.981 ± 0.008 | 0.986 | 0.986 |
| Gradient Boosting | 0.981 ± 0.007 | 0.995 | 0.995 |
| Logistic Regression | 0.969 ± 0.008 | 0.970 | 0.970 |
| K-Nearest Neighbors | 0.961 ± 0.007 | 0.970 | 0.970 |

## Robustness
Accuracy of the selected model on the test set after perturbing the inputs (5 random repeats per setting).

| Gaussian noise (fraction of each feature's std) | Test accuracy |
|---|---|
| 0 | 0.998 ± 0.000 |
| 0.1 | 0.961 ± 0.008 |
| 0.25 | 0.825 ± 0.012 |
| 0.5 | 0.540 ± 0.016 |
| 1 | 0.308 ± 0.019 |

| Share of inputs replaced by the training median | Test accuracy |
|---|---|
| 0% | 0.998 ± 0.000 |
| 10% | 0.905 ± 0.006 |
| 20% | 0.804 ± 0.014 |
| 30% | 0.703 ± 0.017 |

## Calibration
- Expected calibration error (ECE, 10 bins, top-1 confidence): **0.056** (0 is perfect).
- Log loss: **0.067**.
- The "High / Medium / Low" confidence labels in the app are simple thresholds (80% and 50%) on the top probability. They are only as trustworthy as the calibration above.

## Uncertainty: conformal prediction sets
- **Method:** split conformal prediction with the score 1 - p(true crop), calibrated on 440 held-out rows. For each input the app returns every crop whose probability is at least 1 - q (0.180). When no crop reaches that bar, the app shows the three best guesses instead; adding crops can only raise coverage, so the guarantee is kept.
- **Target coverage:** 90%. **Empirical coverage on the test set: 1.000.** Average set size: 1.18 crops; 91% of test inputs get a single crop, and 9% have no crop above the bar and fall back to three guesses.
- The guarantee is marginal (averaged over inputs) and assumes new data resemble the calibration data. It can fail under distribution shift, and coverage for individual crops can differ.

## Out-of-distribution check
- Mahalanobis distance to the pooled training data; inputs beyond the 99% training quantile (distance 4.37) are flagged. 1.1% of test rows are flagged.
- The training data forms one cluster per crop, so this pooled check only catches inputs far from the whole cloud. Per-input range warnings are separate.

## Feature importance (permutation, test set)
| Input | Drop in accuracy when shuffled |
|---|---|
| Humidity | 0.309 |
| Nitrogen (N) | 0.234 |
| Rainfall | 0.170 |
| Potassium (K) | 0.153 |
| Phosphorus (P) | 0.120 |
| Temperature | 0.011 |
| Soil pH | 0.011 |

## Explanations in the app
For each recommendation the app shows how much the probability of the recommended crop would change if one input were replaced by the dataset median. This is a simple occlusion method. It can understate the effect of inputs that are correlated with other inputs.

## Limitations
- The dataset is small, balanced and unusually clean, so scores will not transfer to real fields.
- Only seven inputs are used. Soil type, location, season, irrigation, prices and pests are unknown to the model.
- Inputs outside the training ranges above trigger a warning, and predictions there are unreliable.

## Reproducibility
- Python 3.11.5, scikit-learn 1.9.1, pandas 3.1.1, numpy 2.4.6.
- Re-run `python train.py` on the same CSV to regenerate every number in this file.
