"""Training, evaluation, explanation and prediction helpers for the crop recommendation model."""
from __future__ import annotations

import json
import math
import platform
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score, log_loss
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"

# Hard limits used to reject impossible input. The ranges seen in the training
# data are stored separately and only produce a warning.
FEATURE_INFO = {
    "N": {"label": "Nitrogen (N)", "unit": "ratio in soil", "min": 0, "max": 300, "step": 1},
    "P": {"label": "Phosphorus (P)", "unit": "ratio in soil", "min": 0, "max": 300, "step": 1},
    "K": {"label": "Potassium (K)", "unit": "ratio in soil", "min": 0, "max": 400, "step": 1},
    "temperature": {"label": "Temperature", "unit": "°C", "min": 0, "max": 60, "step": 0.1},
    "humidity": {"label": "Humidity", "unit": "%", "min": 0, "max": 100, "step": 0.1},
    "ph": {"label": "Soil pH", "unit": "pH scale", "min": 0, "max": 14, "step": 0.1},
    "rainfall": {"label": "Rainfall", "unit": "mm", "min": 0, "max": 500, "step": 0.1},
}

DISPLAY_NAMES = {
    "kidneybeans": "Kidney beans",
    "pigeonpeas": "Pigeon peas",
    "mothbeans": "Moth beans",
    "mungbean": "Mung bean",
    "blackgram": "Black gram",
}

NOISE_LEVELS = (0.0, 0.1, 0.25, 0.5, 1.0)  # fraction of each feature's standard deviation
MISSING_FRACTIONS = (0.0, 0.1, 0.2, 0.3)   # share of input values replaced by the training median
EFFECT_THRESHOLD = 0.005                   # 0.5 percentage points


CROP_ICONS = {
    "rice": "🌾", "maize": "🌽", "chickpea": "🫘", "kidneybeans": "🫘", "blackgram": "🫘", "lentil": "🫘",
    "apple": "🍎", "banana": "🍌", "mango": "🥭", "grapes": "🍇", "watermelon": "🍉", "muskmelon": "🍈",
    "orange": "🍊", "papaya": "🍈", "coconut": "🥥", "coffee": "☕", "cotton": "☁️",
}


def crop_icon(crop: str) -> str:
    return CROP_ICONS.get(crop, "🌱")


def display_name(crop: str) -> str:
    return DISPLAY_NAMES.get(crop, crop.replace("_", " ").title())


# ----------------------------------------------------------------- data
def load_dataset(path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Download 'Crop_recommendation.csv' (the Kaggle 'Crop Recommendation Dataset') "
            "and save it as data/Crop_recommendation.csv"
        )
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}. Found: {list(df.columns)}")
    df = df[FEATURES + [TARGET]].dropna().copy()
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="raise")
    df[TARGET] = df[TARGET].astype(str).str.strip().str.lower()
    return df.reset_index(drop=True)


def environment_info() -> dict:
    return {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }


# ------------------------------------------------------------ training
def candidate_models(seed: int = 42) -> dict:
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "K-Nearest Neighbors": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Gaussian Naive Bayes": make_pipeline(GaussianNB()),
        "Decision Tree": make_pipeline(DecisionTreeClassifier(random_state=seed)),
        "Random Forest": make_pipeline(RandomForestClassifier(n_estimators=200, random_state=seed)),
        "Gradient Boosting": make_pipeline(GradientBoostingClassifier(random_state=seed)),
    }


def tuned_candidate_models(seed: int = 42) -> dict:
    """Same candidates, but Random Forest, KNN and Logistic Regression tune their hyperparameters.

    Each tuned model is a RandomizedSearchCV with an inner 3-fold search. Cross-validating such an
    estimator is therefore *nested* cross-validation, so the CV scores are not inflated by the tuning.
    """
    models = candidate_models(seed)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    spaces = {
        "Random Forest": {
            "randomforestclassifier__n_estimators": [100, 200, 400],
            "randomforestclassifier__max_depth": [None, 8, 16, 32],
            "randomforestclassifier__min_samples_leaf": [1, 2, 4],
            "randomforestclassifier__max_features": ["sqrt", "log2"],
        },
        "K-Nearest Neighbors": {
            "kneighborsclassifier__n_neighbors": [3, 5, 7, 9, 15],
            "kneighborsclassifier__weights": ["uniform", "distance"],
        },
        "Logistic Regression": {"logisticregression__C": [0.1, 1.0, 10.0, 100.0]},
    }
    for name, space in spaces.items():
        size = math.prod(len(values) for values in space.values())
        models[name] = RandomizedSearchCV(
            models[name], space, n_iter=min(10, size), cv=inner, scoring="accuracy", random_state=seed
        )
    return models


def _jsonable_params(params: dict) -> dict:
    clean = {}
    for key, value in params.items():
        value = value.item() if hasattr(value, "item") else value
        clean[key.split("__")[-1]] = value if value is None or isinstance(value, (int, float, str, bool)) else str(value)
    return clean


def train_and_evaluate(
    df: pd.DataFrame,
    models_dir="models",
    reports_dir="reports",
    seed: int = 42,
    test_size: float = 0.2,
    calibration_size: float = 0.2,
    alpha: float = 0.1,
    cv_folds: int = 5,
    candidates: dict | None = None,
    dataset_sha256: str | None = None,
    tune: bool = False,
    verbose: bool = False,
) -> dict:
    """Compare candidate models, pick the best by cross-validation, report on a held-out test set.

    Three stratified splits: train (fit and cross-validate), calibration (conformal prediction only)
    and test (used once, for reporting). `test_size` and `calibration_size` are fractions of the whole dataset.
    """
    models_dir, reports_dir = Path(models_dir), Path(reports_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    X, y = df[FEATURES], df[TARGET]
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_dev, y_dev, test_size=calibration_size / (1 - test_size), stratify=y_dev, random_state=seed
    )
    if candidates is None:
        candidates = tuned_candidate_models(seed) if tune else candidate_models(seed)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    comparison, fitted = [], {}
    for name, model in candidates.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        fitted_model = clone(model).fit(X_train, y_train)
        pred = fitted_model.predict(X_test)
        row = {
            "model": name,
            "cv_accuracy_mean": float(scores.mean()),
            "cv_accuracy_std": float(scores.std()),
            "test_accuracy": float(accuracy_score(y_test, pred)),
            "test_macro_f1": float(f1_score(y_test, pred, average="macro")),
        }
        if hasattr(fitted_model, "best_params_"):
            row["best_params"] = _jsonable_params(fitted_model.best_params_)
        comparison.append(row)
        fitted[name] = fitted_model
        if verbose:
            print(f"  {name:<22} cv={row['cv_accuracy_mean']:.3f}±{row['cv_accuracy_std']:.3f}  "
                  f"test={row['test_accuracy']:.3f}  f1={row['test_macro_f1']:.3f}")

    # Choose by cross-validation on the training split only (never by test score).
    best_row = max(comparison, key=lambda r: r["cv_accuracy_mean"])
    best_name = best_row["model"]
    best = fitted[best_name]
    best_pred = best.predict(X_test)

    importance = permutation_importance(
        best, X_test, y_test, n_repeats=10, random_state=seed, scoring="accuracy"
    )
    feature_importance = {f: float(v) for f, v in zip(FEATURES, importance.importances_mean)}

    robustness = evaluate_robustness(best, X_train, X_test, y_test, seed=seed)
    calibration = calibration_report(best, X_test, y_test)
    conformal = conformal_calibrate(best, X_cal, y_cal, X_test, y_test, alpha=alpha)
    ood = fit_ood(X_train, X_test)

    classes = [str(c) for c in best.classes_]
    ranges = {f: [float(df[f].min()), float(df[f].max())] for f in FEATURES}
    baseline = {f: float(df[f].median()) for f in FEATURES}
    profiles = {
        crop: {f: round(float(v), 1) for f, v in row.items()}
        for crop, row in df.groupby(TARGET)[FEATURES].mean().iterrows()
    }
    class_counts = df[TARGET].value_counts()

    metrics = {
        "dataset_rows": int(len(df)),
        "dataset_sha256": dataset_sha256,
        "n_classes": len(classes),
        "class_count_min": int(class_counts.min()),
        "class_count_max": int(class_counts.max()),
        "train_rows": int(len(X_train)),
        "calibration_rows": int(len(X_cal)),
        "test_rows": int(len(X_test)),
        "cv_folds": cv_folds,
        "random_seed": seed,
        "tuned": bool(tune),
        "best_model": best_name,
        "test_accuracy": best_row["test_accuracy"],
        "test_macro_f1": best_row["test_macro_f1"],
        "comparison": comparison,
        "feature_importance": feature_importance,
        "feature_ranges": ranges,
        "robustness": robustness,
        "calibration": calibration,
        "conformal": conformal,
        "ood": {k: ood[k] for k in ("threshold", "quantile", "test_flag_rate")},
        "environment": environment_info(),
    }

    artifact = {
        "model": best,
        "features": FEATURES,
        "classes": classes,
        "ranges": ranges,
        "baseline": baseline,
        "profiles": profiles,
        "conformal": {"alpha": alpha, "qhat": conformal["qhat"]},
        "ood": ood,
        "metrics": metrics,
    }
    joblib.dump(artifact, models_dir / "crop_model.joblib")
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (reports_dir / "model_comparison.md").write_text(_comparison_markdown(metrics), encoding="utf-8")
    (reports_dir / "model_card.md").write_text(model_card_markdown(metrics), encoding="utf-8")
    _plot_confusion_matrix(y_test, best_pred, classes, reports_dir / "confusion_matrix.png")
    _plot_importance(feature_importance, reports_dir / "feature_importance.png")
    _plot_robustness(robustness, reports_dir / "robustness.png")
    _plot_calibration(calibration, reports_dir / "calibration.png")
    return artifact


# ---------------------------------------------------- robustness / calibration
def evaluate_robustness(model, X_train, X_test, y_test, seed: int = 42, repeats: int = 5) -> dict:
    """Test accuracy when inputs are noisy or partly missing (missing values -> training median)."""
    rng = np.random.default_rng(seed)
    std = X_train.std().to_numpy(dtype=float)
    fill = X_train.median().to_numpy(dtype=float)
    lows = np.array([FEATURE_INFO[f]["min"] for f in FEATURES], dtype=float)
    highs = np.array([FEATURE_INFO[f]["max"] for f in FEATURES], dtype=float)
    base = X_test.to_numpy(dtype=float)

    def score(array):
        return float(accuracy_score(y_test, model.predict(pd.DataFrame(array, columns=FEATURES))))

    noise_rows = []
    for level in NOISE_LEVELS:
        scores = [score(base)]
        if level > 0:
            scores = [
                score(np.clip(base + rng.normal(size=base.shape) * std * level, lows, highs))
                for _ in range(repeats)
            ]
        noise_rows.append({"level": level, "accuracy": float(np.mean(scores)), "std": float(np.std(scores))})

    missing_rows = []
    for fraction in MISSING_FRACTIONS:
        scores = [score(base)]
        if fraction > 0:
            scores = [
                score(np.where(rng.random(base.shape) < fraction, fill, base))
                for _ in range(repeats)
            ]
        missing_rows.append({"fraction": fraction, "accuracy": float(np.mean(scores)), "std": float(np.std(scores))})

    return {"noise": noise_rows, "missing": missing_rows, "repeats": repeats}


def calibration_report(model, X_test, y_test, n_bins: int = 10) -> dict:
    """Is the model right about as often as its confidence says? (top-1 confidence vs accuracy)"""
    proba = model.predict_proba(X_test)
    classes = np.array(model.classes_)
    confidence = proba.max(axis=1)
    correct = classes[proba.argmax(axis=1)] == y_test.to_numpy()

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        if not in_bin.any():
            continue
        accuracy = float(correct[in_bin].mean())
        mean_conf = float(confidence[in_bin].mean())
        ece += float(in_bin.mean()) * abs(accuracy - mean_conf)
        bins.append({"confidence": mean_conf, "accuracy": accuracy, "count": int(in_bin.sum())})
    return {
        "ece": float(ece),
        "log_loss": float(log_loss(y_test, proba, labels=list(model.classes_))),
        "bins": bins,
    }


# ------------------------------------------------ conformal prediction / OOD
def prediction_sets(proba: np.ndarray, qhat: float, fallback_k: int = 3) -> np.ndarray:
    """Boolean matrix: which classes are in each prediction set.

    A class is kept when its probability reaches 1 - qhat. When no class does (the model is unsure),
    the top `fallback_k` classes are shown instead of a misleading single guess. Adding classes can only
    raise coverage, so the 1 - alpha guarantee is kept.
    """
    bar = (1.0 - qhat) - 1e-12
    mask = proba >= bar
    rows = np.arange(len(proba))
    mask[rows, proba.argmax(axis=1)] = True
    unsure = proba.max(axis=1) < bar
    if unsure.any():
        top = np.argsort(-proba, axis=1)[:, :fallback_k]
        for row in np.flatnonzero(unsure):
            mask[row, top[row]] = True
    return mask


def conformal_calibrate(model, X_cal, y_cal, X_test, y_test, alpha: float = 0.1) -> dict:
    """Split conformal prediction with the score 1 - p(true class).

    The resulting sets contain the true class with probability >= 1 - alpha (marginally, assuming the
    new data are exchangeable with the calibration data). Coverage is then measured on the test set.
    """
    index = {c: i for i, c in enumerate(model.classes_)}
    proba_cal = model.predict_proba(X_cal)
    true_cal = np.array([index[c] for c in y_cal])
    scores = 1.0 - proba_cal[np.arange(len(true_cal)), true_cal]
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    qhat = float(np.sort(scores)[k - 1]) if k <= n else 1.0  # 1.0 -> every class is included

    proba_test = model.predict_proba(X_test)
    sets = prediction_sets(proba_test, qhat)
    true_test = np.array([index[c] for c in y_test])
    covered = sets[np.arange(len(true_test)), true_test]
    sizes = sets.sum(axis=1)
    return {
        "alpha": float(alpha),
        "target_coverage": float(1 - alpha),
        "n_calibration": int(n),
        "qhat": qhat,
        "empirical_coverage": float(covered.mean()),
        "mean_set_size": float(sizes.mean()),
        "single_crop_fraction": float((sizes == 1).mean()),
        "fallback_fraction": float((proba_test.max(axis=1) < (1.0 - qhat) - 1e-12).mean()),
    }


def fit_ood(X_train, X_test, quantile: float = 0.99) -> dict:
    """Mahalanobis distance to the pooled training data; flag inputs beyond the training quantile."""
    data = X_train.to_numpy(dtype=float)
    mean = data.mean(axis=0)
    covariance = np.cov(data, rowvar=False) + 1e-6 * np.eye(data.shape[1])
    precision = np.linalg.inv(covariance)

    def distances(array):
        diff = array - mean
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", diff, precision, diff), 0.0))

    threshold = float(np.quantile(distances(data), quantile))
    flag_rate = float((distances(X_test.to_numpy(dtype=float)) > threshold).mean())
    return {
        "mean": mean.tolist(),
        "precision": precision.tolist(),
        "threshold": threshold,
        "quantile": quantile,
        "test_flag_rate": flag_rate,
    }


# ----------------------------------------------------------- reports
def _comparison_markdown(metrics: dict) -> str:
    lines = [
        "| Model | CV accuracy (mean ± std) | Test accuracy | Test macro F1 |",
        "|---|---|---|---|",
    ]
    for r in sorted(metrics["comparison"], key=lambda r: -r["cv_accuracy_mean"]):
        mark = " **(selected)**" if r["model"] == metrics["best_model"] else ""
        lines.append(
            f"| {r['model']}{mark} | {r['cv_accuracy_mean']:.3f} ± {r['cv_accuracy_std']:.3f} "
            f"| {r['test_accuracy']:.3f} | {r['test_macro_f1']:.3f} |"
        )
    header = (
        f"Dataset: {metrics['dataset_rows']} rows, {metrics['n_classes']} crops. "
        f"Split: {metrics['train_rows']} train / {metrics['calibration_rows']} calibration / {metrics['test_rows']} test "
        f"(stratified, seed {metrics['random_seed']}). "
        f"Cross-validation: {metrics['cv_folds']}-fold on the training split"
        f"{' (nested: hyperparameters tuned inside each fold)' if metrics.get('tuned') else ''}.\n\n"
    )
    params = [
        f"- {r['model']}: " + ", ".join(f"{k}={v}" for k, v in r["best_params"].items())
        for r in metrics["comparison"] if r.get("best_params")
    ]
    tuned_block = "\n\nSelected hyperparameters (fit on the training split):\n" + "\n".join(params) + "\n" if params else ""
    return header + "\n".join(lines) + "\n" + tuned_block


def model_card_markdown(m: dict) -> str:
    noise = "\n".join(
        f"| {r['level']:g} | {r['accuracy']:.3f} ± {r['std']:.3f} |" for r in m["robustness"]["noise"]
    )
    missing = "\n".join(
        f"| {r['fraction']:.0%} | {r['accuracy']:.3f} ± {r['std']:.3f} |" for r in m["robustness"]["missing"]
    )
    importance = "\n".join(
        f"| {FEATURE_INFO[f]['label']} | {v:.3f} |"
        for f, v in sorted(m["feature_importance"].items(), key=lambda kv: -kv[1])
    )
    ranges = "\n".join(
        f"| {FEATURE_INFO[f]['label']} | {lo:g} to {hi:g} |" for f, (lo, hi) in m["feature_ranges"].items()
    )
    env = m["environment"]
    cal = m["calibration"]
    con = m["conformal"]
    ood = m["ood"]
    sha = m.get("dataset_sha256") or "not recorded"
    tuned_note = "; Random Forest, KNN and Logistic Regression were tuned inside each fold (nested cross-validation)" if m.get("tuned") else ""
    per_crop = (
        f"{m['class_count_min']} examples"
        if m["class_count_min"] == m["class_count_max"]
        else f"{m['class_count_min']} to {m['class_count_max']} examples"
    )
    return f"""# Model card: crop recommendation classifier

## Overview
- **Task:** multi-class classification. Given seven soil and climate measurements, rank the {m['n_classes']} crops by suitability.
- **Selected model:** {m['best_model']} (chosen by {m['cv_folds']}-fold cross-validation on the training split{tuned_note}).
- **Intended use:** education and demonstration of an ML decision-support workflow.
- **Out of scope:** yield forecasts, fertilizer prescriptions, or any decision about real land without local expert advice.

## Data
- {m['dataset_rows']} rows, {m['n_classes']} crops, {per_crop} per crop.
- SHA-256 of the CSV used: `{sha}`

| Input | Range in training data |
|---|---|
{ranges}

## Training and evaluation
- Stratified split: {m['train_rows']} train / {m['calibration_rows']} calibration / {m['test_rows']} test, seed {m['random_seed']}. The calibration split is used only for the conformal prediction sets below.
- The test set is used once, to report the selected model. It is never used to choose between models.
- **Held-out test accuracy: {m['test_accuracy']:.3f}. Macro F1: {m['test_macro_f1']:.3f}.**

{_comparison_markdown(m).split(chr(10) + chr(10), 1)[1]}
## Robustness
Accuracy of the selected model on the test set after perturbing the inputs ({m['robustness']['repeats']} random repeats per setting).

| Gaussian noise (fraction of each feature's std) | Test accuracy |
|---|---|
{noise}

| Share of inputs replaced by the training median | Test accuracy |
|---|---|
{missing}

## Calibration
- Expected calibration error (ECE, 10 bins, top-1 confidence): **{cal['ece']:.3f}** (0 is perfect).
- Log loss: **{cal['log_loss']:.3f}**.
- The "High / Medium / Low" confidence labels in the app are simple thresholds (80% and 50%) on the top probability. They are only as trustworthy as the calibration above.

## Uncertainty: conformal prediction sets
- **Method:** split conformal prediction with the score 1 - p(true crop), calibrated on {con['n_calibration']} held-out rows. For each input the app returns every crop whose probability is at least 1 - q ({con['qhat']:.3f}). When no crop reaches that bar, the app shows the three best guesses instead; adding crops can only raise coverage, so the guarantee is kept.
- **Target coverage:** {con['target_coverage']:.0%}. **Empirical coverage on the test set: {con['empirical_coverage']:.3f}.** Average set size: {con['mean_set_size']:.2f} crops; {con['single_crop_fraction']:.0%} of test inputs get a single crop, and {con['fallback_fraction']:.0%} have no crop above the bar and fall back to three guesses.
- The guarantee is marginal (averaged over inputs) and assumes new data resemble the calibration data. It can fail under distribution shift, and coverage for individual crops can differ.

## Out-of-distribution check
- Mahalanobis distance to the pooled training data; inputs beyond the {ood['quantile']:.0%} training quantile (distance {ood['threshold']:.2f}) are flagged. {ood['test_flag_rate']:.1%} of test rows are flagged.
- The training data forms one cluster per crop, so this pooled check only catches inputs far from the whole cloud. Per-input range warnings are separate.

## Feature importance (permutation, test set)
| Input | Drop in accuracy when shuffled |
|---|---|
{importance}

## Explanations in the app
For each recommendation the app shows how much the probability of the recommended crop would change if one input were replaced by the dataset median. This is a simple occlusion method. It can understate the effect of inputs that are correlated with other inputs.

## Limitations
- The dataset is small, balanced and unusually clean, so scores will not transfer to real fields.
- Only seven inputs are used. Soil type, location, season, irrigation, prices and pests are unknown to the model.
- Inputs outside the training ranges above trigger a warning, and predictions there are unreliable.

## Reproducibility
- Python {env['python']}, scikit-learn {env['scikit_learn']}, pandas {env['pandas']}, numpy {env['numpy']}.
- Re-run `python train.py` on the same CSV to regenerate every number in this file.
"""


def _plot_confusion_matrix(y_true, y_pred, classes, path):
    fig, ax = plt.subplots(figsize=(10, 9))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=classes, display_labels=[display_name(c) for c in classes],
        xticks_rotation=90, include_values=False, cmap="Greens", ax=ax, colorbar=False,
    )
    ax.set_title("Confusion matrix (held-out test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_importance(importance: dict, path):
    items = sorted(importance.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([FEATURE_INFO[k]["label"] for k, _ in items], [v for _, v in items], color="#2e7d32")
    ax.set_xlabel("Drop in test accuracy when the feature is shuffled")
    ax.set_title("Permutation feature importance")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_robustness(robustness: dict, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    noise, missing = robustness["noise"], robustness["missing"]
    axes[0].errorbar(
        [r["level"] for r in noise], [r["accuracy"] for r in noise],
        yerr=[r["std"] for r in noise], marker="o", color="#2e7d32", capsize=3,
    )
    axes[0].set_xlabel("Gaussian noise (fraction of each feature's std)")
    axes[0].set_ylabel("Test accuracy")
    axes[0].set_title("Noisy inputs")
    axes[1].errorbar(
        [r["fraction"] * 100 for r in missing], [r["accuracy"] for r in missing],
        yerr=[r["std"] for r in missing], marker="o", color="#c62828", capsize=3,
    )
    axes[1].set_xlabel("Inputs replaced by the training median (%)")
    axes[1].set_title("Missing inputs")
    for ax in axes:
        ax.set_ylim(0, 1.03)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_calibration(calibration: dict, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly calibrated")
    bins = calibration["bins"]
    ax.plot(
        [b["confidence"] for b in bins], [b["accuracy"] for b in bins],
        "o-", color="#2e7d32", label=f"Model (ECE {calibration['ece']:.3f})",
    )
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Fraction correct")
    ax.set_title("Reliability diagram (test set)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------- inference
def load_artifact(path):
    return joblib.load(path)


def validate_input(raw) -> tuple[dict, dict]:
    """Return (clean values, errors). `raw` maps feature name -> string or number."""
    values, errors = {}, {}
    for name in FEATURES:
        info = FEATURE_INFO[name]
        value = raw.get(name) if hasattr(raw, "get") else None
        if value is None or (isinstance(value, str) and not value.strip()):
            errors[name] = f"{info['label']} is required."
            continue
        if isinstance(value, bool):
            errors[name] = f"{info['label']} must be a number."
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            errors[name] = f"{info['label']} must be a number."
            continue
        if not math.isfinite(number):
            errors[name] = f"{info['label']} must be a finite number."
        elif not info["min"] <= number <= info["max"]:
            errors[name] = f"{info['label']} must be between {info['min']} and {info['max']}."
        else:
            values[name] = number
    return values, errors


def range_warnings(artifact: dict, values: dict) -> list[str]:
    warnings = []
    for name, number in values.items():
        lo, hi = artifact["ranges"][name]
        if number < lo or number > hi:
            warnings.append(
                f"{FEATURE_INFO[name]['label']} is outside the range seen in the training data "
                f"({lo:g} to {hi:g}), so the prediction may be unreliable."
            )
    return warnings


def _probabilities(artifact: dict, values: dict) -> tuple[list, np.ndarray]:
    frame = pd.DataFrame([[values[f] for f in artifact["features"]]], columns=artifact["features"])
    return list(artifact["model"].classes_), artifact["model"].predict_proba(frame)[0]


def predict(artifact: dict, values: dict, top_k: int = 3) -> list[dict]:
    classes, probabilities = _probabilities(artifact, values)
    ranked = sorted(zip(classes, probabilities), key=lambda cp: -cp[1])[:top_k]
    return [
        {"crop": str(c), "name": display_name(str(c)), "probability": float(p)}
        for c, p in ranked
    ]


def plausible_crops(artifact: dict, values: dict) -> list[dict]:
    """Conformal prediction set: crops that, with the target coverage, include the right one."""
    classes, probabilities = _probabilities(artifact, values)
    mask = prediction_sets(probabilities[None, :], artifact["conformal"]["qhat"])[0]
    chosen = sorted(((classes[i], probabilities[i]) for i in np.flatnonzero(mask)), key=lambda cp: -cp[1])
    return [{"crop": str(c), "name": display_name(str(c)), "probability": float(p)} for c, p in chosen]


def ood_check(artifact: dict, values: dict) -> dict | None:
    ood = artifact.get("ood")
    if not ood:
        return None
    diff = np.array([values[f] for f in artifact["features"]], dtype=float) - np.array(ood["mean"])
    distance = float(np.sqrt(max(diff @ np.array(ood["precision"]) @ diff, 0.0)))
    return {"distance": distance, "threshold": ood["threshold"], "flagged": distance > ood["threshold"]}


def explain(artifact: dict, values: dict, crop: str) -> list[dict]:
    """Occlusion explanation: how much does P(crop) change if one input is set to the dataset median?

    A positive effect means the user's value supports the recommendation.
    """
    features, baseline = artifact["features"], artifact["baseline"]
    rows = [[values[f] for f in features]]
    for feature in features:
        changed = dict(values)
        changed[feature] = baseline[feature]
        rows.append([changed[f] for f in features])
    probabilities = artifact["model"].predict_proba(pd.DataFrame(rows, columns=features))
    index = list(artifact["model"].classes_).index(crop)
    full = float(probabilities[0][index])

    result = []
    for position, feature in enumerate(features, start=1):
        effect = full - float(probabilities[position][index])
        if effect > EFFECT_THRESHOLD:
            direction = "supports"
        elif effect < -EFFECT_THRESHOLD:
            direction = "lowers"
        else:
            direction = "neutral"
        result.append({
            "feature": feature,
            "label": FEATURE_INFO[feature]["label"],
            "unit": FEATURE_INFO[feature]["unit"],
            "value": values[feature],
            "typical": baseline[feature],
            "effect": effect,
            "direction": direction,
        })
    result.sort(key=lambda r: -abs(r["effect"]))
    return result


def confidence_level(probability: float) -> str:
    if probability >= 0.8:
        return "High"
    if probability >= 0.5:
        return "Medium"
    return "Low"
