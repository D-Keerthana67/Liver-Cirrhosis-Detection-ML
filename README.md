# Liver Cirrhosis Detection using Machine Learning

An end-to-end academic machine learning project for predicting liver cirrhosis status from tabular clinical data. The project compares four supervised learning algorithms: K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Random Forest, and AdaBoost.

## Project Objective

The goal is to build a reproducible ML pipeline that:
- loads and validates a clinical tabular dataset
- handles missing values and categorical features
- encodes categorical variables
- scales numerical features where appropriate
- trains KNN, SVM, Random Forest, and AdaBoost models
- evaluates models using accuracy, precision, recall, F1-score, and confusion matrices
- compares model performance
- saves trained models for later prediction

> **Important:** This is an academic machine-learning project, not a medical diagnostic system. Model performance must not be interpreted as clinical validation or used for patient-care decisions.

## Project Structure

```
Liver-Cirrhosis-Detection-ML/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── src/
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── predict.py
├── models/
│   └── README.md
├── results/
│   └── README.md
└── notebooks/
    └── README.md
```

## Machine Learning Models

1. **KNN** - classifies a sample using nearby training observations.
2. **SVM** - learns a decision boundary that separates classes.
3. **Random Forest** - combines predictions from multiple decision trees.
4. **AdaBoost** - combines weak learners sequentially to improve classification.

## Dataset

Place your CSV dataset inside the `data/` directory. The code accepts a configurable target column, so it is not tied to a particular dataset schema.

Example:

```text
data/
└── liver_data.csv
```

Do not commit private patient information or confidential clinical records.

## Installation

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

From the project root:

```bash
python src/train_models.py --data data/liver_data.csv --target Dataset
```

Replace `Dataset` with the actual target-column name in your dataset.

The script creates:
- `models/*.joblib` for trained models
- `results/model_comparison.csv` for metrics
- `results/confusion_matrices.png` for visual evaluation

## Prediction

After training:

```bash
python src/predict.py --model models/random_forest.joblib --data data/liver_data.csv --row 0
```

The row number is zero-based. Use the same feature schema used during training.

## Evaluation Metrics

The project reports:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

No performance values are hard-coded in this repository; results are generated from the dataset supplied by the user.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

## Future Improvements

- Hyperparameter tuning with cross-validation
- ROC-AUC and precision-recall curves
- Feature importance and model interpretability
- SHAP-based explanations
- Model versioning
- External validation on an independent dataset

## Disclaimer

This repository is intended for educational and research purposes. It does not replace medical professionals, clinical testing, or validated diagnostic tools.
