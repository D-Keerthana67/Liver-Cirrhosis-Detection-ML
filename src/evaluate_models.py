"""Generate detailed evaluation reports and confusion matrices."""

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split

from preprocessing import load_data, split_features_target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()

    data = load_data(args.data)
    features, target = split_features_target(data, args.target)

    _, x_test, _, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )

    model_dir = Path("models")
    result_dir = Path("results")
    result_dir.mkdir(exist_ok=True)

    model_names = ["knn", "svm", "random_forest", "adaboost"]

    for name in model_names:
        model_path = model_dir / f"{name}.joblib"

        if not model_path.exists():
            print(f"Skipping {name}: {model_path} not found.")
            continue

        model = joblib.load(model_path)
        predictions = model.predict(x_test)

        print(f"\n===== {name.upper()} =====")
        print(classification_report(y_test, predictions, zero_division=0))

        display = ConfusionMatrixDisplay.from_predictions(
            y_test, predictions, xticks_rotation="vertical"
        )
        display.ax_.set_title(f"{name.replace('_', ' ').title()} Confusion Matrix")
        plt.tight_layout()
        plt.savefig(result_dir / f"{name}_confusion_matrix.png", dpi=150)
        plt.close()

    print("\nEvaluation completed. Check the results/ directory.")


if __name__ == "__main__":
    main()
