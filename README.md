# Project: fraud-detection-ieee

The framing matters. The README and code should not read like a Kaggle competition entry. It should read like something built by someone who understands why fraud detection is hard operationally — class imbalance, concept drift, the cost asymmetry between false negatives and false positives, the need to explain decisions to compliance. That's what ING's Financial Crime team deals with every day.

What makes this project different from a generic ML repo
Generic Kaggle projectThis projectMaximise AUCExplicit cost-sensitive evaluation (missing fraud costs more than blocking a good transaction)Single modelModel comparison with business justification for final choiceFeature importanceSHAP with financial crime narrative ("this transaction triggered because...")Train/test splitTemporal split — because fraud patterns shift over time, random splits leak future informationAccuracy/AUC onlyPrecision-Recall curve, F1 at operating threshold, business metrics

# Architecture

fraud_detection/
│
├── fraud_detection/
│   ├── __init__.py
│   ├── dataset.py       # TransactionData  — loads, merges train_transaction + train_identity, temporal split
│   ├── features.py      # FeatureEngineer  — velocity features, aggregations, card behaviour, time features
│   ├── model.py         # FraudModel       — LightGBM + Logistic Regression, cost-sensitive evaluation
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


# What each class does differently from the credit project
TransactionData — the IEEE dataset comes as two files (train_transaction.csv + train_identity.csv) that must be merged on TransactionID. The split will be temporal on TransactionDT (a time delta column), not random — because in production you always train on past data and predict on future data. Random splits in fraud detection are a known methodological error that inflates reported AUC.
FeatureEngineer — this is where the project earns its credibility. Raw IEEE features are mostly anonymised (V1–V339). The real signal comes from engineered features:

Transaction velocity (how many transactions from this card in the last N transactions)
Deviation from card's own history (this amount vs card's typical amount)
Time features (hour of day, day of week — fraud spikes at night and weekends)
Card-email domain pairing consistency
Address match between billing and shipping

FraudModel — evaluation will use Precision-Recall AUC (better than ROC-AUC for severe imbalance) and a cost matrix: false negative (missed fraud) costs ~10x more than false positive (blocked legitimate transaction). The operating threshold will be chosen to minimise expected cost, not set arbitrarily at 0.5.
FraudExplainer — SHAP plus a plain-English alert narrative: "This transaction was flagged because the amount ($2,340) is 8x this card's median transaction, it occurred at 3am, and the billing address does not match the shipping address." That's what a compliance analyst actually reads.

# Features that signal financial crime domain knowledge

Temporal split with explicit justification
Cost matrix instead of symmetric loss
Precision-Recall curve as primary evaluation (with explanation of why, not ROC)
Velocity and behavioural deviation features
Alert narrative in explainer
README framed around AML/fraud prevention context, not "I competed on Kaggle"