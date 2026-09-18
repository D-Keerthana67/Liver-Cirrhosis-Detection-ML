"""Predict the target class for one row from a CSV file."""

import argparse

import joblib
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .joblib model")
    parser.add_argument("--data", required=True, help="Path to CSV data")
    parser.add_argument("--row", type=int, default=0, help="Zero-based row number")
    args = parser.parse_args()

    model = joblib.load(args.model)
    data = pd.read_csv(args.data)

    if args.row < 0 or args.row >= len(data):
        raise IndexError(f"Row must be between 0 and {len(data) - 1}.")

    row = data.iloc[[args.row]].copy()

    prediction = model.predict(row)
    print(f"Predicted class: {prediction[0]}")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(row)[0]
        classes = model.classes_

        print("\nClass probabilities:")
        for class_name, probability in zip(classes, probabilities):
            print(f"{class_name}: {probability:.4f}")


if __name__ == "__main__":
    main()
