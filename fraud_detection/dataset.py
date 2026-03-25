"""
dataset.py
-----------
Loads the data, merges, ans splits 

Check the shape, the fraud rate, the split sizes.

"""

import pandas as pd
import numpy as np
from config import (TRANSACTION_PATH, IDENTITY_PATH, TARGET_COLUMN, TIME_COLUMN, ID_COLUMN, TEMPORAL_SPLIT_RATIO, V_FEATURE_KEEP_THRESHOLD)


class TransactionData:
    """
    Loads, merges, and holds the dataset.
    """

    def __init__(
        self,
        transaction_path = TRANSACTION_PATH,
        identity_path = IDENTITY_PATH
    ):
        self.transaction_path = transaction_path
        self.identity_path = identity_path

        self.df = None
        self.X_train = None
        self.X_test = None

    def load(self):
        transactions_df = pd.read_csv(self.transaction_path)
        print("Transactions loaded.")

        identity_df = pd.read_csv(self.identity_path)
        print("Identity records loaded.")

        df = transactions_df.merge(identity_df, on=ID_COLUMN, how="left")
        print(f"Merged df shape: {df.shape}")

        df = self._clean(df)
        print("Data cleaned")

        df = self._temporal_split(df)
        print("Data splited temporally.")
        
        # print summary

        self._loaded = True
        return self
    
    # def get_split(self):
    #     self._check_loaded()
    #     return 

    # defines how an object is represented as a string, mainly for developers/debugging.
    # When you print an object or inspect it in the console, Python calls __repr__:
    def __repr__(self):
        pass

    # ------------------------------------------
    # PRIVATE
    # ------------------------------------------

    def _clean(self, df):
        """
        1. Drop ID.
        2. Handle columns with excessive missingness.
        3. Downcast numerics.
        """
        df = df.drop(columns = [ID_COLUMN], errors="ignore")

        v_cols = [c for c in df.columns if c.startswith("V")]
        missing_rate = df[v_cols].isnull().mean()
        drop_v = missing_rate[missing_rate > V_FEATURE_KEEP_THRESHOLD].index.tolist()
        df = df.drop(columns = drop_v)
        print(f"Dropped {len(drop_v)} V-features (>{V_FEATURE_KEEP_THRESHOLD:.0%} missing).")
        
        df = self._downcast(df)
        return df

    @staticmethod
    def _downcast(df):
        """ 
        Downcast int64 to int32 and float64 to float 32 
        """
        for col in df.select_dtypes(include="int64").columns:
            df[col] = df[col].astype(np.int32)
        for col in df.select_dtypes(include="float64").columns:
            df[col] = df[col].astype(np.float32)
        return df
    
    def _temporal_split(self, df):
        """ 
        Sort by time and split at TEMPORAL_SPLIT_RATIO.
        """
        df_sorted = df.sort_values(TIME_COLUMN).reset_index(drop=True)
        print(1)
        split_idx = int(len(df_sorted) * TEMPORAL_SPLIT_RATIO)
        self.split_time = df_sorted[TIME_COLUMN].iloc(split_idx)
        print(2)
        train = df_sorted.iloc[:split_idx]
        test = df_sorted.iloc[split_idx:]
        print(3)
        feature_cols = [
            c for c in df_sorted.columns
            if c != TARGET_COLUMN
        ]
        print(4)
        self.feature_names = feature_cols
        self.X_train = train[feature_cols]
        self.X_test = test[feature_cols]
        self.y_train = train[TARGET_COLUMN]
        self.y_test = test[TARGET_COLUMN]


    def _check_loaded(self):
        if not self._loaded:
            raise RuntimeError("Data not loaded.")

        
    # ------------------------------------------
    # OTHER
    # ------------------------------------------

td = TransactionData(TRANSACTION_PATH, IDENTITY_PATH)
td.load()