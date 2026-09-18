# Models

Trained models are generated locally by `src/train_models.py`.

Generated `.joblib` files are ignored by Git so that large or dataset-specific model artifacts are not committed accidentally.

Expected generated files include:

- `knn.joblib`
- `svm.joblib`
- `random_forest.joblib`
- `adaboost.joblib`

Each saved file contains the preprocessing pipeline together with the classifier, allowing the same transformations to be applied during prediction.
