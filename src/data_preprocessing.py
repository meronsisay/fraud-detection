"""
Data preprocessing
"""

import pandas as pd
from pathlib import Path
import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DataPreprocessor:
    """Handle loading, assessment, and cleaning of data tailored for Fraud Detection."""

    def __init__(self, raw_path="data/raw/"):
        self.raw_path = Path(raw_path)
        self.df = None
        self.quality_report = {}

    def load_data(self, filename):
        """Load CSV file with path validation."""
        file_path = self.raw_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Target data file not found at: {file_path}")
        self.df = pd.read_csv(file_path)
        logging.info(f"Loaded: {filename} | Shape: {self.df.shape}")
        return self

    def assess_quality(self):
        """Assess data quality and store metrics."""
        print("\n" + "=" * 60)
        print("DATA QUALITY ASSESSMENT")
        print("=" * 60)

        rows = len(self.df)
        cols = len(self.df.columns)
        print(f"\n BASIC INFO:\n   Rows: {rows:,}\n   Columns: {cols}")

        # Smart duplicate calculation based on dataset schema
        if "user_id" in self.df.columns:
            dup = self.df.duplicated(subset=["user_id"]).sum()
            dup_msg = f"   {dup} rows ({dup/rows*100:.2f}%) based on unique 'user_id'"
        else:
            dup = self.df.duplicated().sum()
            dup_msg = f"   {dup} rows ({dup/rows*100:.2f}%) based on complete row match"
        print(f"\nDUPLICATES:\n{dup_msg}")

        missing = self.df.isnull().sum()
        missing = missing[missing > 0]
        print("\nMISSING VALUES:")
        if len(missing) == 0:
            print("   None")
        else:
            for col in missing.index:
                pct = missing[col] / rows * 100
                print(f"   {col}: {missing[col]:,} ({pct:.1f}%)")

        self.quality_report = {
            "rows": rows,
            "cols": cols,
            "duplicates": dup,
            "missing_cols": len(missing),
        }
        return self

    def fix_data_types(self):
        """Fix incorrect data types cleanly without early category mutation pitfalls."""
        print("\n" + "=" * 60)
        print("FIXING DATA TYPES")
        print("=" * 60)

        # Datetime casting
        for col in ["signup_time", "purchase_time"]:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col])
                print(f"  Converted '{col}' → datetime64[ns]")

        # Explicit target conversion
        for col in ["class", "Class"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(int)
                print(f"  Converted target '{col}' → int")

        return self

    def handle_missing(self):
        """Handle missing values safely avoiding modern pandas deprecation errors."""
        print("\n" + "=" * 60)
        print("HANDLING MISSING VALUES")
        print("=" * 60)

        missing_total = self.df.isnull().sum().sum()
        if missing_total == 0:
            print("No missing values to handle.")
            return self

        # CRITICAL FINTECH RULE: Drop records missing key alignment features (like IP Address)
        if "ip_address" in self.df.columns and self.df["ip_address"].isnull().sum() > 0:
            null_ips = self.df["ip_address"].isnull().sum()
            self.df = self.df.dropna(subset=["ip_address"])
            print(
                f"   DROPPED {null_ips} rows due to missing 'ip_address' (Unmappable Geolocation)"
            )

        # Safe Imputation without using deprecated inplace=True
        for col in self.df.columns:
            missing_count = self.df[col].isnull().sum()
            if missing_count > 0:
                if self.df[col].dtype in ["int64", "float64"]:
                    median_val = self.df[col].median()
                    self.df[col] = self.df[col].fillna(median_val)
                    print(f"   IMPUTED '{col}': median value ({median_val})")
                else:
                    mode_val = self.df[col].mode()[0]
                    self.df[col] = self.df[col].fillna(mode_val)
                    print(f"   IMPUTED '{col}': mode value ('{mode_val}')")
        return self

    def remove_duplicates(self):
        """
        Remove duplicate rows targeting business logic definitions.
        E-commerce requires unique user profiles; Banking requires unique transaction records.
        """
        print("\n" + "=" * 60)
        print("REMOVING DUPLICATES")
        print("=" * 60)

        before = len(self.df)

        # Scenario A: E-commerce data (1 unique record expected per user identity)
        if "user_id" in self.df.columns:
            self.df = self.df.drop_duplicates(subset=["user_id"], keep="first")
            justification = (
                "Dropped duplicate user profiles (user_id should be strictly unique)."
            )

        # Scenario B: Credit card/ledger data (Look for identical transactional footprints)
        else:
            self.df = self.df.drop_duplicates(keep="first")
            justification = (
                "Dropped identical transaction rows (likely API retry errors)."
            )

        removed = before - len(self.df)
        print(f"  Result: {justification}")
        print(
            f"  Removed {removed} duplicate records. Current row count: {len(self.df):,}"
        )
        return self

    def convert_ip_to_int(self):
        """Converts IPv4 string/float addresses to integer format, handling both numeric and dot-decimal formats."""
        if "ip_address" not in self.df.columns:
            return self

        print("\n" + "=" * 60)
        print("CONVERTING IP ADDRESSES TO INTEGERS")
        print("=" * 60)

        def parse_ip(val):
            if pd.isna(val):
                return np.nan
            val_str = str(val).strip()

            # If it's a standard dotted IPv4 string (e.g., '192.168.1.1')
            if "." in val_str and len(val_str.split(".")) == 4:
                try:
                    parts = list(map(int, val_str.split(".")))
                    return (
                        (parts[0] << 24) + (parts[1] << 16) + (parts[2] << 8) + parts[3]
                    )
                except ValueError:
                    return np.nan

            # Fallback for e-commerce float/numeric string formatting
            try:
                return int(float(val_str))
            except ValueError:
                return np.nan

        self.df["ip_address_int"] = self.df["ip_address"].apply(parse_ip)
        print("  Successfully transformed 'ip_address' → 'ip_address_int'")
        return self

    def merge_geolocation(self, ip_map_path="../data/raw/IpAddress_to_Country.csv"):
        """Performs optimized interval matching using pandas merge_asof."""
        print("\n" + "=" * 60)
        print("GEOLOCATION INTEGRATION VIA RANGE LOOKUP")
        print("=" * 60)

        # Load and clean map boundaries inline
        ip_map = pd.read_csv(ip_map_path).dropna().drop_duplicates()
        ip_map["lower_bound_ip_address"] = (
            ip_map["lower_bound_ip_address"].astype(float).astype(int)
        )
        ip_map["upper_bound_ip_address"] = (
            ip_map["upper_bound_ip_address"].astype(float).astype(int)
        )

        # Sort keys is a strict algorithmic prerequisite for merge_asof
        self.df = self.df.sort_values("ip_address_int")
        ip_map = ip_map.sort_values("lower_bound_ip_address")

        # Backward match on lower bound
        merged = pd.merge_asof(
            self.df,
            ip_map,
            left_on="ip_address_int",
            right_on="lower_bound_ip_address",
            direction="backward",
        )

        # Strict logic validation: clean matches leaking past upper bound bounds
        merged["country"] = merged.apply(
            lambda row: (
                row["country"]
                if row["ip_address_int"] <= row["upper_bound_ip_address"]
                else "Unknown"
            ),
            axis=1,
        )

        # Drop the intermediate boundary range mapping columns to keep dataframe memory footprint lean
        self.df = merged.drop(
            columns=["lower_bound_ip_address", "upper_bound_ip_address"]
        )
        print(
            f"  Merged country data. Identified {self.df['country'].nunique()} unique nations."
        )
        return self

    def engineer_features(self):
        """Extracts deep temporal patterns and user behavioral velocity metrics."""
        print("\n" + "=" * 60)
        print("ENGINEERING TEMPORAL AND BEHAVIORAL FEATURES")
        print("=" * 60)

        if "purchase_time" in self.df.columns and "signup_time" in self.df.columns:
            # 1. Delta Time Feature
            self.df["time_since_signup"] = (
                self.df["purchase_time"] - self.df["signup_time"]
            ).dt.total_seconds() / 3600.0

            # 2. Cyclic Temporal Extraction
            self.df["hour_of_day"] = self.df["purchase_time"].dt.hour
            self.df["day_of_week"] = self.df["purchase_time"].dt.dayofweek
            print(
                "  Created: 'time_since_signup' (hours), 'hour_of_day', 'day_of_week'"
            )

        if "device_id" in self.df.columns:
            # 3. Behavioral Velocity Feature (Tracks multi-account device exploitation attempts)
            self.df["device_tx_velocity"] = self.df.groupby("device_id")[
                "device_id"
            ].transform("count")
            print(
                "  Created: 'device_tx_velocity' (Frequency metric mapping device recurrence patterns)"
            )

        return self
    
    def engineer_advanced_features(self):
            """Leak-free features based directly on EDA findings."""
            # 1. Early transaction risk (Flags the 3.3x higher risk window discovered in EDA)
            if 'time_since_signup' in self.df.columns:
                # Your engineer_features computes this in hours, so <= 4 is perfect!
                self.df['is_first_4hours'] = (self.df['time_since_signup'] <= 4).astype(int)
            
            # 2. High-risk countries (Clusters the ~26% extreme fraud rate regions safely)
            if 'country' in self.df.columns:
                high_risk_nations = {'Ecuador', 'Tunisia', 'Peru'}
                self.df['is_high_risk_country'] = self.df['country'].isin(high_risk_nations).astype(int)
                
            return self

    def save(self, output_path):
        """Save cleaned data to processed folder destination."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(out, index=False)
        logging.info(f"Saved cleanly processed data to: {output_path}")
        return self

    def get_data(self):
        return self.df
