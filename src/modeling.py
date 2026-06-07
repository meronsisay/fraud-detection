"""
Modular data preparation for fraud detection
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter


class ModelingPreparer:
    def __init__(self, target_col="class"):
        self.target_col = target_col
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(
            drop="first", sparse_output=False, handle_unknown="ignore"
        )
        self.before_counts = None
        self.after_counts = None

    def prepare_splits(self, df: pd.DataFrame):
        """Splits, encodes, and scales features safely."""

        # Drop non-feature columns
        cols_to_drop = [
            self.target_col,
            "user_id",
            "device_id",
            "ip_address",
            "signup_time",
            "purchase_time",
            "ip_address_int",
        ]
        X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        y = df[self.target_col]

        # Identify column types
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Encode categorical
        if cat_cols:
            encoded_train = self.encoder.fit_transform(X_train[cat_cols])
            encoded_test = self.encoder.transform(X_test[cat_cols])
            encoded_cols = self.encoder.get_feature_names_out(cat_cols)

            df_encoded_train = pd.DataFrame(
                encoded_train, columns=encoded_cols, index=X_train.index
            )
            df_encoded_test = pd.DataFrame(
                encoded_test, columns=encoded_cols, index=X_test.index
            )

            X_train = X_train.drop(columns=cat_cols).join(df_encoded_train)
            X_test = X_test.drop(columns=cat_cols).join(df_encoded_test)

        # Scale numerical
        if num_cols:
            X_train[num_cols] = self.scaler.fit_transform(X_train[num_cols])
            X_test[num_cols] = self.scaler.transform(X_test[num_cols])

        return X_train, X_test, y_train, y_test

    def apply_resampling(self, X_train, y_train, strategy="smote", target_ratio=None):
        """
        Apply resampling based on chosen strategy and optional custom target ratio.

        Parameters:
        -----------
        strategy : str, ('none', 'undersample', 'oversample', 'smote')
        target_ratio : float, optional (e.g., 0.25 means minority class will become 25% of majority class)
        """
        self.before_counts = Counter(y_train)

        if strategy == "none":
            self.after_counts = self.before_counts
            return X_train, y_train

        # If target_ratio is None, imblearn defaults to a 1:1 balance (1.0)
        strat_param = target_ratio if target_ratio is not None else "auto"

        if strategy == "undersample":
            sampler = RandomUnderSampler(sampling_strategy=strat_param, random_state=42)
        elif strategy == "oversample":
            sampler = RandomOverSampler(sampling_strategy=strat_param, random_state=42)
        elif strategy == "smote":
            sampler = SMOTE(sampling_strategy=strat_param, random_state=42)
        else:
            raise ValueError(
                "Strategy must be 'none', 'undersample', 'oversample', or 'smote'"
            )

        X_res, y_res = sampler.fit_resample(X_train, y_train)
        self.after_counts = Counter(y_res)

        return X_res, y_res

    def get_imbalance_report(self):
        """Return concise imbalance report"""
        if self.before_counts is None:
            return None

        legit_before = self.before_counts[0]
        fraud_before = self.before_counts[1]
        legit_after = self.after_counts[0]
        fraud_after = self.after_counts[1]

        report = {
            "before": {
                "legit": legit_before,
                "fraud": fraud_before,
                "ratio": f"1:{legit_before/fraud_before:.1f}",
            },
            "after": {
                "legit": legit_after,
                "fraud": fraud_after,
                "ratio": f"1:{legit_after/fraud_after:.1f}",
            },
        }

        return report
