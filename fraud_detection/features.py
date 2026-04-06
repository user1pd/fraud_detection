"""
features.py
-----------------
Creates additional fraud-predictive features from the raw IEEE-CIS columns.

The signal could come from:

1. Behavioral deviation features

2. Velocity features

3. Time features

4. Consistency features
    Keep those with meaningful variance.
-----------------
Categorical encoding
We use target encoding for high-cardinality categoricals.

"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from config import (
    TARGET_COLUMN,
    TIME_COLUMN,

)

class FeatureEngineer:
    """
        Engineers fraud-predictive features from raw transaction data

        Usage
        -----
        engineer = FeatureEngineer()
        X_train_eng.fit_transform(X_train, y_train)
        X_test_eng.transform(X_test)
    """

    def __init__(self):
        self._scaler = StandardScaler()

        self._cols_to_scale = None
        self._feature_names_out = None
        self._is_fitted = False
    
    def fit_tranform(self):
        pass

    def transform(self):
        pass

    def __repr__(self):
        pass