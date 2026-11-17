Dataset: 2200 rows, 22 crops. Split: 1320 train / 440 calibration / 440 test (stratified, seed 42). Cross-validation: 5-fold on the training split.

| Model | CV accuracy (mean ± std) | Test accuracy | Test macro F1 |
|---|---|---|---|
| Random Forest **(selected)** | 0.997 ± 0.002 | 0.998 | 0.998 |
| Gaussian Naive Bayes | 0.996 ± 0.002 | 0.995 | 0.995 |
| Decision Tree | 0.981 ± 0.008 | 0.986 | 0.986 |
| Gradient Boosting | 0.981 ± 0.007 | 0.995 | 0.995 |
| Logistic Regression | 0.969 ± 0.008 | 0.970 | 0.970 |
| K-Nearest Neighbors | 0.961 ± 0.007 | 0.970 | 0.970 |
