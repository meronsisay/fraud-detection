"""
EDA functions for fraud detection datasets
"""

import pandas as pd
import matplotlib.pyplot as plt

class EDAUtils:
    """EDA tools for any fraud dataset"""

    def __init__(self, df, name="Dataset", target_col="class"):
        self.df = df
        self.name = name
        self.target = target_col if target_col in df.columns else None

    def class_imbalance(self):
        """Quantify class imbalance"""
        if self.target is None:
            print(f" Target column '{self.target}' not found")
            return None

        counts = self.df[self.target].value_counts()
        percentages = self.df[self.target].value_counts(normalize=True) * 100

        print("\n" + "=" * 50)
        print(f"CLASS IMBALANCE: {self.name}")
        print("=" * 50)
        print(f"Class 0: {counts.get(0, 0):,} ({percentages.get(0, 0):.4f}%)")
        print(f"Class 1: {counts.get(1, 0):,} ({percentages.get(1, 0):.4f}%)")

        if len(counts) == 2:
            ratio = counts[0] / counts[1]
            print(f"Imbalance Ratio: 1:{ratio:.1f}")

        return counts, percentages

    def plot_class_distribution(self):
            """Plot class distribution with explicit data labels"""
            if self.target is None:
                return

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # 1. Fetch exact value counts
            counts = self.df[self.target].value_counts()
            
            # 2. Bar plot with explicit label injections
            counts.plot(
                kind="bar", ax=ax1, color=["green", "red"]
            )
            ax1.set_title(f"{self.name} - Class Distribution", fontsize=12, fontweight='bold')
            ax1.set_xlabel("Class", fontsize=10)
            ax1.set_ylabel("Count", fontsize=10)
            ax1.set_xticklabels(["Legit", "Fraud"], rotation=0)
            
            # Automatically inject numeric count text on top of each bar
            for container in ax1.containers:
                ax1.bar_label(container, fmt='{:,}', padding=3, fontweight='bold')

            # 3. Pie chart with clean percentage mapping
            counts.plot(
                kind="pie",
                ax=ax2,
                autopct="%1.2f%%",
                labels=["Legit", "Fraud"], # Explicitly label pieces
                colors=["green", "red"],
                explode=[0, 0.15], # Increase explode slightly so tiny fraud slices stand out
                startangle=90,      # Better structural alignment
                textprops={'weight': 'bold'} # Makes percentage texts easier to read
            )
            ax2.set_title("Class Proportion", fontsize=12, fontweight='bold')
            ax2.set_ylabel("")

            plt.tight_layout()
            plt.show()

    def numerical_summary(self, numerical_cols):
        """Summary statistics using describe()"""
        print("\n" + "=" * 50)
        print(f"NUMERICAL FEATURES: {self.name}")
        print("=" * 50)

        # Filter existing columns
        existing_cols = [col for col in numerical_cols if col in self.df.columns]
        if existing_cols:
            print(self.df[existing_cols].describe().T)
        else:
            print("No numerical columns found")

    def categorical_summary(self, categorical_cols):
        """Summary for categorical features using value_counts"""
        print("\n" + "=" * 50)
        print(f"CATEGORICAL FEATURES: {self.name}")
        print("=" * 50)

        for col in categorical_cols:
            if col in self.df.columns:
                print(f"\n{col}:")
                print(self.df[col].value_counts().head(5).to_string())

    def plot_numerical_by_target(self, numerical_cols):
        """Plot numerical feature distributions by target class"""
        if self.target is None:
            return

        n_cols = min(len(numerical_cols), 4)
        if n_cols == 0:
            return

        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
        if n_cols == 1:
            axes = [axes]

        for i, col in enumerate(numerical_cols[:4]):
            for label in [0, 1]:
                subset = self.df[self.df[self.target] == label][col]
                axes[i].hist(
                    subset, bins=50, alpha=0.5, label=f"Class {label}", density=True
                )
            axes[i].set_title(f"{col}")
            axes[i].legend()

        plt.tight_layout()
        plt.show()

    def plot_categorical_by_target(self, categorical_cols):
        """Plot fraud rate by categorical feature"""
        if self.target is None:
            return

        n_cols = min(len(categorical_cols), 3)
        if n_cols == 0:
            return

        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
        if n_cols == 1:
            axes = [axes]

        for i, col in enumerate(categorical_cols[:3]):
            fraud_rate = (
                self.df.groupby(col)[self.target].mean().sort_values(ascending=False)
            )
            fraud_rate.head(10).plot(kind="bar", ax=axes[i], color="red")
            axes[i].set_title(f"Fraud Rate by {col}")
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Fraud Rate")
            axes[i].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()

    def correlation_with_target(self, numerical_cols):
        """Show correlation with target"""
        if self.target is None:
            return

        correlations = []
        for col in numerical_cols:
            if col in self.df.columns:
                corr = self.df[col].corr(self.df[self.target])
                correlations.append({"feature": col, "correlation": corr})

        corr_df = pd.DataFrame(correlations).sort_values(
            "correlation", key=abs, ascending=False
        )

        print("\n" + "=" * 50)
        print(f"CORRELATION WITH TARGET: {self.name}")
        print("=" * 50)
        for _, row in corr_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['correlation']:.4f}")

        return corr_df
