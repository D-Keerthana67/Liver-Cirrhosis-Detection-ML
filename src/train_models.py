"""Train and compare KNN, SVM, Random Forest, and AdaBoost models."""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from preprocessing import build_preprocessor, load_data, split_features_target


def build_models():
    """Return the four classifiers used in the project."""
    return {
        "knn": KNeighborsClassifier(n_neighbors=5),
        "svm": SVC(kernel="rbf", probability=True, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=42
        ),
        "adaboost": AdaBoostClassifier(n_estimators=100, random_state=42),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--test-size", type=float, default=0.20)
    args = parser.parse_args()

    data = load_data(args.data)
    features, target = split_features_target(data, args.target)

    if target.nunique() < 2:
        raise ValueError("The target column must contain at least two classes.")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=args.test_size,
        random_state=42,
        stratify=target,
    )

    preprocessor = build_preprocessor(features)
    models = build_models()

    model_dir = Path("models")
    result_dir = Path("results")
    model_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)

    rows = []

    for name, classifier in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )

        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)

        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(
                    y_test, predictions, average="weighted", zero_division=0
                ),
                "recall": recall_score(
                    y_test, predictions, average="weighted", zero_division=0
                ),
                "f1_score": f1_score(
                    y_test, predictions, average="weighted", zero_division=0
                ),
            }
        )

        joblib.dump(pipeline, model_dir / f"{name}.joblib")

    results = pd.DataFrame(rows)
    results.to_csv(result_dir / "model_comparison.csv", index=False)

    print("\nModel comparison:")
    print(results.to_string(index=False))
    print("\nSaved trained models to models/")
    print("Saved metrics to results/model_comparison.csv")


if __name__ == "__main__":
    main()
