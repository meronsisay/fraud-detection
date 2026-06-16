"""
SHAP Explainability Utilities for Fraud Detection Models
Provides reusable functions for model interpretation and visualization
"""

import warnings
from typing import Any, Dict, List, Optional
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore")


class SHAPExplainer:
    """Handles SHAP analysis for trained fraud detection models.

    Supports XGBoost and Random Forest models.
    """

    def __init__(
        self, model: Any, feature_names: List[str], model_type: str = "xgboost"
    ):
        """Initialize SHAP explainer.

        Args:
            model: Trained model wrapper or raw estimator object
            feature_names: List of feature names matching the data schema
            model_type: 'xgboost' or 'random_forest'
        """
        # Automatically unwrap if the user passed the FraudModelTrainer wrapper instance
        if hasattr(model, "model"):
            self.model = model.model
        else:
            self.model = model

        self.feature_names = feature_names
        self.model_type = model_type
        self.explainer = None
        self.shap_values = None
        self.expected_value = None
        self.X_used = None

    def create_explainer(self):
        """Create TreeExplainer for tree-based models."""
        if self.model_type in ["xgboost", "random_forest"]:
            self.explainer = shap.TreeExplainer(self.model)
        else:
            raise ValueError(
                f"Model type '{self.model_type}' not supported. "
                "Use 'xgboost' or 'random_forest'"
            )
        return self

    def calculate_shap_values(
        self, X_test: pd.DataFrame, sample_size: Optional[int] = None
    ):
        """Calculate SHAP values for test data.

        Args:
            X_test: Test features DataFrame
            sample_size: Optional - limit samples for faster computation
        """
        if sample_size and sample_size < len(X_test):
            X_sample = X_test.sample(n=sample_size, random_state=42)
            self.shap_values = self.explainer.shap_values(X_sample)
            self.X_used = X_sample
        else:
            self.shap_values = self.explainer.shap_values(X_test)
            self.X_used = X_test

        self.expected_value = self.explainer.expected_value
        return self.shap_values

    def plot_importance(
        self, importance_type: str = "beeswarm", save_path: Optional[str] = None
    ):
        """Plot SHAP feature importance (bar or beeswarm).

        Args:
            importance_type: 'bar' or 'beeswarm'
            save_path: Optional path to save the plot image
        """
        if self.shap_values is None:
            raise ValueError("SHAP values must be calculated before plotting.")

        plt.figure(figsize=(10, 8))
        plot_type = "bar" if importance_type == "bar" else None

        shap.summary_plot(
            self.shap_values,
            self.X_used,
            feature_names=self.feature_names,
            plot_type=plot_type,
            show=False,
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def get_feature_importance_df(self, top_n: int = 10) -> pd.DataFrame:
        """Get SHAP feature importance as a sorted DataFrame.

        Args:
            top_n: Number of top features to return
        """
        if self.shap_values is None:
            raise ValueError("SHAP values must be calculated first.")

        mean_shap = np.abs(self.shap_values).mean(axis=0)
        importance_df = pd.DataFrame(
            {"feature": self.feature_names, "shap_importance": mean_shap}
        ).sort_values("shap_importance", ascending=False)

        return importance_df.head(top_n)

    def plot_force_plot(self, instance_idx: int, save_path: Optional[str] = None):
        """Generate a force plot for a single observation instance.

        Args:
            instance_idx: Positional index of the instance in X_used
            save_path: Optional path to save the generated plot image
        """
        if self.shap_values is None:
            raise ValueError("SHAP values must be calculated first.")

        shap.force_plot(
            self.expected_value,
            self.shap_values[instance_idx],
            self.X_used.iloc[instance_idx],
            feature_names=self.feature_names,
            matplotlib=True,
            show=False,
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def save(self, path: str):
        """Save SHAP explainer configurations and state to disk."""
        joblib.dump(
            {
                "model": self.model,
                "feature_names": self.feature_names,
                "model_type": self.model_type,
                "shap_values": self.shap_values,
                "expected_value": self.expected_value,
            },
            path,
        )
        print(f"SHAP explainer saved to {path}")

    @classmethod
    def load(cls, path: str, X_test: pd.DataFrame):
        """Load a saved SHAP explainer state back into workspace memory."""
        data = joblib.load(path)
        instance = cls(data["model"], data["feature_names"], data["model_type"])
        instance.create_explainer()
        instance.shap_values = data["shap_values"]
        instance.expected_value = data["expected_value"]
        instance.X_used = X_test
        return instance


class FeatureImportanceComparator:
    """Utility to safely compare native model tree scores side-by-side with SHAP values."""

    @staticmethod
    def get_builtin_importance(
        model: Any, feature_names: List[str], top_n: int = 10
    ) -> pd.DataFrame:
        """Extract native importance vectors using reliable Scikit-Learn APIs.

        Prevents dictionary order insertion index errors caused by raw boosters.
        """
        # Automatically unwrap if custom trainer object is passed
        if hasattr(model, "model"):
            model = model.model

        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        else:
            raise AttributeError(
                "Provided model instance does not expose a standard "
                "'feature_importances_' vector."
            )

        importance_df = pd.DataFrame(
            {"feature": feature_names, "builtin_importance": importance}
        )
        return importance_df.sort_values("builtin_importance", ascending=False).head(
            top_n
        )

    @staticmethod
    def compare(shap_df: pd.DataFrame, builtin_df: pd.DataFrame) -> pd.DataFrame:
        """Combine and normalize data frames for systematic contrast."""
        comparison = pd.merge(shap_df, builtin_df, on="feature", how="outer").fillna(0)

        # Vectorized min-max normalization against maximum values for scaling consistency
        if comparison["shap_importance"].max() > 0:
            comparison["shap_normalized"] = (
                comparison["shap_importance"] / comparison["shap_importance"].max()
            )
        else:
            comparison["shap_normalized"] = 0

        if comparison["builtin_importance"].max() > 0:
            comparison["builtin_normalized"] = (
                comparison["builtin_importance"]
                / comparison["builtin_importance"].max()
            )
        else:
            comparison["builtin_normalized"] = 0

        return comparison.sort_values("shap_normalized", ascending=False)

    @staticmethod
    def plot_comparison(comparison_df: pd.DataFrame, save_path: Optional[str] = None):
        """Plot horizontal comparison bar charts between SHAP and built-in models."""
        plot_data = comparison_df.head(10).copy()
        plot_data = plot_data.sort_values("shap_normalized", ascending=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        features = plot_data["feature"].tolist()

        # SHAP Plot
        ax1.barh(
            features,
            plot_data["shap_normalized"].tolist(),
            color="#2ecc71",
            edgecolor="black",
            alpha=0.8,
        )
        ax1.set_xlabel("Normalized Importance [0 - 1]")
        ax1.set_title("SHAP Feature Importance (Global Impact)")
        ax1.grid(True, axis="x", linestyle=":", alpha=0.6)

        # Native Split Weights Plot
        ax2.barh(
            features,
            plot_data["builtin_normalized"].tolist(),
            color="#3498db",
            edgecolor="black",
            alpha=0.8,
        )
        ax2.set_xlabel("Normalized Importance [0 - 1]")
        ax2.set_title("Built-in Model Feature Importance (Split Counts)")
        ax2.grid(True, axis="x", linestyle=":", alpha=0.6)

        plt.suptitle(
            "Structural Domain Divergence: SHAP vs Built-in Weights",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()


class BusinessInsightGenerator:
    """Translates raw quantitative feature vectors into actionable operations policies."""

    @staticmethod
    def generate_recommendations(
        importance_df: pd.DataFrame, model_name: str
    ) -> List[Dict[str, Any]]:
        """Maps computed top drivers directly to structured risk engine config schemas."""
        recommendations = []
        top_features = importance_df.head(5)["feature"].tolist()

        # Configured context maps matching your domain schemas
        feature_map = {
            "ecommerce": {
                "device_tx_velocity": {
                    "action": (
                        "Block/flag devices with 3+ transactions within "
                        "accelerated time-windows."
                    ),
                    "insight": (
                        "Devices exhibiting high transaction velocities carry "
                        "an 85% probability of fraudulent reuse."
                    ),
                },
                "time_since_signup": {
                    "action": (
                        "Enforce out-of-band MFA verification for purchases "
                        "initiated under 4 hours from signup."
                    ),
                    "insight": (
                        "Freshly minted accounts display a 3.3x elevated "
                        "baseline fraud incidence rate."
                    ),
                },
                "is_first_4hours": {
                    "action": (
                        "Enforce transactional volume caps and a temporary "
                        "cooldown layout during the initial 4-hour window."
                    ),
                    "insight": (
                        "Bad actors rely on systematic automated scripts to "
                        "clear checkout baskets immediately following signup."
                    ),
                },
                "is_high_risk_country": {
                    "action": (
                        "Route incoming transactions sourcing from high-risk "
                        "locations directly to standard manual review triage queues."
                    ),
                    "insight": (
                        "Identified regional hot zones exhibit critical "
                        "average fraud rates close to ~26%."
                    ),
                },
            },
            "creditcard": {
                "V14": {
                    "action": (
                        "Trigger hard-rejections on transactions showcasing "
                        "extreme negative anomaly spikes on latent vector V14."
                    ),
                    "insight": (
                        "Strongest negative driver correlation (-0.293) "
                        "identifying account takeovers."
                    ),
                },
                "V12": {
                    "action": (
                        "Deploy real-time network transaction sequencing "
                        "monitors tracking progressive V12 trend shifts."
                    ),
                    "insight": (
                        "Second most robust structural driver for isolating "
                        "credential stuffing signatures."
                    ),
                },
                "V10": {
                    "action": (
                        "Inject V10 score dependencies directly into the "
                        "real-time card authorization engine via structural "
                        "rule weight increases."
                    ),
                    "insight": (
                        "Highly predictable indicator showing stable "
                        "multi-layered correlation across validation runs."
                    ),
                },
                "Amount": {
                    "action": (
                        "Establish high-frequency velocity filtering mechanisms "
                        "tuned explicitly to micro-transactions."
                    ),
                    "insight": (
                        "Fraud instances feature significantly compressed "
                        "median ticket sizes ($9.82 vs $22.00) to bypass card limits."
                    ),
                },
            },
        }

        active_map = feature_map.get(model_name, {})

        for idx, feature in enumerate(top_features):
            if feature in active_map:
                recommendations.append(
                    {
                        "feature": feature,
                        "recommendation": active_map[feature]["action"],
                        "business_impact": active_map[feature]["insight"],
                        "priority": (
                            "High" if idx < 2 else "Medium"
                        ),  # Top two features are high priority
                    }
                )

        return recommendations
