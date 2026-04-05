"""
config.py
=========

Source of all cpath, constants, and hyperparameters.

"""

import os

# -------------------------------------
# PATHS
# -------------------------------------

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")

TRANSACTION_PATH = os.path.join(DATA_DIR, "train_transaction.csv")
IDENTITY_PATH = os.path.join(DATA_DIR, "train_identity.csv")

# print("file:", __file__)
# print("abspath:", os.path.abspath(__file__))
# print("dirname:", os.path.dirname(os.path.abspath(__file__)))
# print("->>>:", DATA_DIR)
# print("->>>:", os.path.join(ROOT_DIR, "data", "x", "y"))

# -------------------------------------
# DATA
# -------------------------------------

TARGET_COLUMN = "isFraud"
TIME_COLUMN = "TransactionDT"
ID_COLUMN = "TransactionID"

TEMPORAL_SPLIT_RATIO = 0.75

CARD_COLUMNS = ["card1", "card2", "card3", "card4", "card5", "card6"]
ADDR_COLUMNS = ["addr1", "addr2"]
EMAIL_COLUMNS = ["P_emaildomain", "R_emaildomain"] # purchaser and recipient email domain


# TransactionAMT: transaction payment amount in USD
# ProductCD: product code, the product for each transaction
# card1 - card6: payment card information, such as card type, card category, issue bank, country, etc.
# addr: address
# dist: distance
# P_ and (R__) emaildomain: purchaser and recipient email domain
# C1-C14: counting, such as how many addresses are found to be associated with the payment card, etc. The actual meaning is masked.
# D1-D15: timedelta, such as days between previous transaction, etc.
# M1-M9: match, such as names on card and address, etc.
# Vxxx: Vesta engineered rich features, including ranking, counting, and other entity relations.
# Categorical Features: ProductCD card1 - card6 addr1, addr2 P_emaildomain R_emaildomain M1 - M9
V_FEATURE_KEEP_THRESHOLD = 0.5   # drop V-features with >50% missing

# -------------------------------------
# FEATURE ENGINEERING
# -------------------------------------




# -------------------------------------
# MODELS
# -------------------------------------

RANDOM_STATE = 48


# -------------------------------------
# EVALUATION and COST MATRIX
# -------------------------------------

PRIMARY_METRIC = "pr-auc"

# -------------------------------------
# EXPLAINER
# -------------------------------------

SHAP_SAMPLE_SIZE = 2000
TOP_N_FEATURES = 20