"""
Data preprocessing 
"""

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        print("\n" + "="*60)
        print("DATA QUALITY ASSESSMENT")
        print("="*60)
        
        rows = len(self.df)
        cols = len(self.df.columns)
        print(f"\n BASIC INFO:\n   Rows: {rows:,}\n   Columns: {cols}")
        
        # Smart duplicate calculation based on dataset schema
        if 'user_id' in self.df.columns:
            dup = self.df.duplicated(subset=['user_id']).sum()
            dup_msg = f"   {dup} rows ({dup/rows*100:.2f}%) based on unique 'user_id'"
        else:
            dup = self.df.duplicated().sum()
            dup_msg = f"   {dup} rows ({dup/rows*100:.2f}%) based on complete row match"
        print(f"\nDUPLICATES:\n{dup_msg}")
        
        missing = self.df.isnull().sum()
        missing = missing[missing > 0]
        print(f"\nMISSING VALUES:")
        if len(missing) == 0:
            print("   None")
        else:
            for col in missing.index:
                pct = missing[col] / rows * 100
                print(f"   {col}: {missing[col]:,} ({pct:.1f}%)")
        
        self.quality_report = {
            'rows': rows,
            'cols': cols,
            'duplicates': dup,
            'missing_cols': len(missing)
        }
        return self
    
    def fix_data_types(self):
        """Fix incorrect data types cleanly without early category mutation pitfalls."""
        print("\n" + "="*60)
        print("FIXING DATA TYPES")
        print("="*60)
        
        # Datetime casting
        for col in ['signup_time', 'purchase_time']:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col])
                print(f"  Converted '{col}' → datetime64[ns]")
        
        # Explicit target conversion
        for col in ['class', 'Class']:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(int)
                print(f"  Converted target '{col}' → int")
                
        return self
    
    def handle_missing(self):
        """Handle missing values safely avoiding modern pandas deprecation errors."""
        print("\n" + "="*60)
        print("HANDLING MISSING VALUES")
        print("="*60)
        
        missing_total = self.df.isnull().sum().sum()
        if missing_total == 0:
            print("No missing values to handle.")
            return self
            
        # CRITICAL FINTECH RULE: Drop records missing key alignment features (like IP Address)
        if 'ip_address' in self.df.columns and self.df['ip_address'].isnull().sum() > 0:
            null_ips = self.df['ip_address'].isnull().sum()
            self.df = self.df.dropna(subset=['ip_address'])
            print(f"   DROPPED {null_ips} rows due to missing 'ip_address' (Unmappable Geolocation)")
            
        # Safe Imputation without using deprecated inplace=True
        for col in self.df.columns:
            missing_count = self.df[col].isnull().sum()
            if missing_count > 0:
                if self.df[col].dtype in ['int64', 'float64']:
                    median_val = self.df[col].median()
                    self.df[col] = self.df[col].fillna(median_val)
                    print(f"   IMPUTED '{col}': median value ({median_val})")
                else:
                    mode_val = self.df[col].mode()[0]
                    self.df[col] = self.df[col].fillna(mode_val)
                    print(f"   IMPUTED '{col}': mode value ('{mode_val}')")
        return self
    
    def remove_duplicates(self):
        """Remove duplicate rows targeting business logic definitions."""
        print("\n" + "="*60)
        print("REMOVING DUPLICATES")
        print("="*60)
        
        before = len(self.df)
        if 'user_id' in self.df.columns:
            self.df = self.df.drop_duplicates(subset=['user_id'], keep='first')
        else:
            self.df = self.df.drop_duplicates(keep='first')
            
        removed = before - len(self.df)
        print(f"Removed {removed} duplicate records. Current row count: {len(self.df):,}")
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

