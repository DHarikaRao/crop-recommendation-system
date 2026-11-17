"""Train and evaluate the crop recommendation model.

Usage:
    python train.py                              # uses data/Crop_recommendation.csv
    python train.py --data path/to/file.csv
    python train.py --tune                       # nested-CV hyperparameter tuning (slower)

Writes models/crop_model.joblib and reports/ (metrics, charts and a model card).
"""
import argparse
import hashlib
import sys
from pathlib import Path

from crop_model import load_dataset, train_and_evaluate


def sha256_of(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--data", default="data/Crop_recommendation.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tune", action="store_true",
                        help="tune Random Forest, KNN and Logistic Regression with nested cross-validation (slower)")
    args = parser.parse_args()

    try:
        df = load_dataset(args.data)
    except (FileNotFoundError, ValueError) as error:
        print(f"\n❌ {error}")
        sys.exit(1)

    print(f"Loaded {len(df)} rows, {df['label'].nunique()} crops. Training (this can take a few minutes)...\n")
    artifact = train_and_evaluate(
        df, seed=args.seed, dataset_sha256=sha256_of(args.data), tune=args.tune, verbose=True
    )
    m = artifact["metrics"]
    noisy = m["robustness"]["noise"][2]
    print(f"\n✅ Selected: {m['best_model']}  |  test accuracy {m['test_accuracy']:.3f}  |  macro F1 {m['test_macro_f1']:.3f}")
    print(f"   Accuracy with noise at {noisy['level']:g} std: {noisy['accuracy']:.3f}  |  calibration error (ECE): {m['calibration']['ece']:.3f}")
    c = m["conformal"]
    print(f"   Conformal sets: {c['empirical_coverage']:.3f} coverage (target {c['target_coverage']:.2f}), "
          f"{c['mean_set_size']:.2f} crops per set on average")
    print("Saved: models/crop_model.joblib and reports/ (metrics.json, model_card.md, charts)")


if __name__ == "__main__":
    main()
