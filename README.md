# Project: fraud-detection-ieee


# Architecture

fraud_detection/
│
├── fraud_detection/
│   ├── __init__.py
│   ├── dataset.py       # TransactionData  — loads, merges train_transaction + train_identity, temporal split
│   ├── features.py      # FeatureEngineer  — velocity features, aggregations, card behaviour, time features
│   ├── model.py         # FraudModel       — LightGBM + Logistic Regression, cost-sensitive evaluation
|   ├── evaluation.py
│   ├── explainer.py     # FraudExplainer   — SHAP, alert narrative generation
│   └── pipeline.py      # FraudPipeline    — orchestrates everything
│
├── notebooks/
│   └── analysis.ipynb   # EDA + class imbalance exploration + model walkthrough
│
├── data/raw/            # Kaggle CSVs here
├── outputs/
├── config.py
├── main.py
├── requirements.txt
└── README.md

