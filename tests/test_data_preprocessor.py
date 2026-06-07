"""
Tests for DataPreprocessor class
"""

import pytest
import pandas as pd
from pathlib import Path
from src.data_preprocessing import DataPreprocessor


class TestDataPreprocessor:
    """Test cases for DataPreprocessor"""

    def test_initialization(self):
        """Test that preprocessor initializes correctly"""
        prep = DataPreprocessor(raw_path="data/raw/")
        assert prep.raw_path == Path("data/raw/")
        assert prep.df is None
        assert prep.quality_report == {}

    def test_load_data_file_not_found(self):
        """Test that loading non-existent file raises error"""
        prep = DataPreprocessor(raw_path="data/raw/")
        with pytest.raises(FileNotFoundError):
            prep.load_data("nonexistent.csv")

    def test_assess_quality(self):
        """Test quality assessment on sample data"""
        prep = DataPreprocessor()
        prep.df = pd.DataFrame(
            {"user_id": [1, 2, 3], "amount": [100, 200, 300], "class": [0, 1, 0]}
        )

        result = prep.assess_quality()
        assert result.quality_report["rows"] == 3
        assert result.quality_report["cols"] == 3

    def test_remove_duplicates_with_user_id(self):
        """Test duplicate removal for e-commerce data"""
        prep = DataPreprocessor()
        prep.df = pd.DataFrame(
            {"user_id": [1, 1, 2, 3], "amount": [100, 100, 200, 300]}
        )

        prep.remove_duplicates()
        assert len(prep.df) == 3  # Duplicate user_id 1 removed

    def test_remove_duplicates_without_user_id(self):
        """Test duplicate removal for credit card data"""
        prep = DataPreprocessor()
        prep.df = pd.DataFrame(
            {
                "V1": [1, 1, 2, 3],
                "V2": [1, 1, 2, 3],
                "Amount": [100, 100, 200, 300],
                "Class": [0, 0, 1, 0],
            }
        )

        prep.remove_duplicates()
        assert len(prep.df) == 3  # Duplicate row removed

    def test_fix_data_types_datetime(self):
        """Test datetime conversion"""
        prep = DataPreprocessor()
        prep.df = pd.DataFrame(
            {
                "signup_time": ["2024-01-01 10:00:00", "2024-01-02 11:00:00"],
                "purchase_time": ["2024-01-01 10:05:00", "2024-01-02 11:05:00"],
                "class": [0, 1],
            }
        )

        prep.fix_data_types()
        assert pd.api.types.is_datetime64_any_dtype(prep.df["signup_time"])
        assert pd.api.types.is_datetime64_any_dtype(prep.df["purchase_time"])

    def test_fix_data_types_target(self):
        """Test target column conversion"""
        prep = DataPreprocessor()
        prep.df = pd.DataFrame({"class": ["0", "1", "0"], "amount": [100, 200, 300]})

        prep.fix_data_types()
        assert prep.df["class"].dtype == "int64"

    def test_handle_missing_no_missing(self):
        """Test handling when no missing values"""
        prep = DataPreprocessor()
        prep.df = pd.DataFrame({"amount": [100, 200, 300], "class": [0, 1, 0]})

        result = prep.handle_missing()
        assert len(result.df) == 3

    def test_handle_missing_with_ip(self):
        """Test dropping rows with missing IP addresses"""
        prep = DataPreprocessor()
        prep.df = pd.DataFrame(
            {"ip_address": ["1.1.1.1", None, "3.3.3.3"], "amount": [100, 200, 300]}
        )

        prep.handle_missing()
        assert len(prep.df) == 2  # Row with None dropped
        assert prep.df["ip_address"].iloc[0] == "1.1.1.1"

    def test_convert_ip_to_int(self):
        """Test IP address to integer conversion"""
        prep = DataPreprocessor()
        prep.df = pd.DataFrame(
            {"ip_address": ["192.168.1.1", "10.0.0.1", "52093.496895"]}
        )

        prep.convert_ip_to_int()
        assert "ip_address_int" in prep.df.columns

        # 192.168.1.1 correctly converts to 3232235777
        assert prep.df["ip_address_int"].iloc[0] == 3232235777

        # 10.0.0.1 correctly converts to 167772161
        assert prep.df["ip_address_int"].iloc[1] == 167772161

        # 52093.496895 truncates as a fallback to 52093
        assert prep.df["ip_address_int"].iloc[2] == 52093

    def test_save_data(self, tmp_path):
        """Test saving data to file"""
        prep = DataPreprocessor()
        prep.df = pd.DataFrame({"col1": [1, 2, 3]})

        output_path = tmp_path / "test_output.csv"
        prep.save(str(output_path))

        assert output_path.exists()
        saved_df = pd.read_csv(output_path)
        assert len(saved_df) == 3

    def test_get_data(self):
        """Test getting dataframe"""
        prep = DataPreprocessor()
        prep.df = pd.DataFrame({"col1": [1, 2, 3]})

        df = prep.get_data()
        assert df.equals(prep.df)
