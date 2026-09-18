# Dataset

Place the liver cirrhosis/liver disease CSV dataset in this directory.

Example:

```text
data/
└── liver_data.csv
```

The training script accepts the dataset path and target-column name as command-line arguments.

Example:

```bash
python src/train_models.py --data data/liver_data.csv --target Dataset
```

## Data privacy

Do not upload private patient information, personally identifiable information, hospital records, or confidential clinical datasets to this public repository.

The repository intentionally contains no clinical dataset and no fabricated results.
