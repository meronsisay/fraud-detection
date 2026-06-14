"""
Modular data preparation and model experimentation
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)


class ModelingPreparer:
    """Handles data preparation: splitting, encoding, scaling, resampling"""

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

        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        all_nums = X.select_dtypes(include=[np.number]).columns.tolist()

        binary_flags = ["is_first_4hours", "is_high_risk_country"]
        num_cols = [col for col in all_nums if col not in binary_flags]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

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

        if num_cols:
            X_train[num_cols] = self.scaler.fit_transform(X_train[num_cols])
            X_test[num_cols] = self.scaler.transform(X_test[num_cols])

        return X_train, X_test, y_train, y_test

    def apply_resampling(self, X_train, y_train, strategy="smote", target_ratio=None):
        """Apply resampling based on chosen strategy"""
        self.before_counts = Counter(y_train)

        if strategy == "none":
            self.after_counts = self.before_counts
            return X_train, y_train

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

        legit_before, fraud_before = self.before_counts[0], self.before_counts[1]
        legit_after, fraud_after = self.after_counts[0], self.after_counts[1]

        return {
            "before": {
                "legit": legit_before,
                "fraud": fraud_before,
                "ratio": f"1:{legit_before/max(1, fraud_before):.1f}",
            },
            "after": {
                "legit": legit_after,
                "fraud": fraud_after,
                "ratio": f"1:{legit_after/max(1, fraud_after):.1f}",
            },
        }

    def save_pipeline(self, path):
        """Save fitted scaler and encoder"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(
            {
                "scaler": self.scaler,
                "encoder": self.encoder,
                "target_col": self.target_col,
            },
            path,
        )
        print(f"Pipeline saved to {path}")

    def load_pipeline(self, path):
        """Load fitted scaler and encoder"""
        pipeline_artifacts = joblib.load(path)
        self.scaler = pipeline_artifacts["scaler"]
        self.encoder = pipeline_artifacts["encoder"]
        self.target_col = pipeline_artifacts["target_col"]
        return self


class FraudModelTrainer:
    """Train and evaluate fraud detection models"""

    def __init__(
        self,
        model_type="xgboost",
        random_state=42,
        scale_pos_weight=1,
        model_params=None,
    ):
        self.model_type = model_type
        self.random_state = random_state
        self.scale_pos_weight = scale_pos_weight
        self.model_params = model_params if model_params is not None else {}
        self.model = self._initialize_model()
        self.results = {}

    def _initialize_model(self):
        """Initialize model based on type, feeding custom grid search params smoothly"""
        if self.model_type == "logistic":
            params = {
                "class_weight": "balanced",
                "random_state": self.random_state,
                "max_iter": 1000,
            }
            params.update(self.model_params)
            return LogisticRegression(**params)

        elif self.model_type == "random_forest":
            params = {
                "n_estimators": 100,
                "max_depth": 10,
                "class_weight": "balanced",
                "random_state": self.random_state,
                "n_jobs": -1,
            }
            params.update(self.model_params)
            return RandomForestClassifier(**params)

        elif self.model_type == "xgboost":
            params = {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "scale_pos_weight": self.scale_pos_weight,
                "random_state": self.random_state,
                "use_label_encoder": False,
                "eval_metric": "logloss",
                "n_jobs": -1,
            }
            params.update(self.model_params)
            return XGBClassifier(**params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_test, y_test, threshold=0.30):
        """Evaluate model performance applying a custom probability threshold"""
        y_proba = self.predict_proba(X_test)
        y_pred = (y_proba >= threshold).astype(int)

        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)

        self.results = {
            "f1_score": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred),
            "accuracy": accuracy_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "auprc": auc(recall_vals, precision_vals),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }
        return self.results

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.model = joblib.load(path)
        return self


# ============================================
# EXPERIMENT & TUNING FUNCTIONS
# ============================================


def run_experiment(
    df,
    target_col,
    strategy="smote",
    target_ratio=None,
    model_type="xgboost",
    threshold=0.30,
    model_params=None,
):
    """Run a single experiment with specified resampling strategy and parameters"""
    preparer = ModelingPreparer(target_col=target_col)
    X_train, X_test, y_train, y_test = preparer.prepare_splits(df)

    if strategy == "none" and model_type == "xgboost":
        counts = y_train.value_counts()
        calculated_weight = counts.get(0, 1) / counts.get(1, 1)
    else:
        calculated_weight = 1

    X_train_res, y_train_res = preparer.apply_resampling(
        X_train, y_train, strategy=strategy, target_ratio=target_ratio
    )

    trainer = FraudModelTrainer(
        model_type=model_type,
        scale_pos_weight=calculated_weight,
        model_params=model_params,
    )
    trainer.train(X_train_res, y_train_res)
    results = trainer.evaluate(X_test, y_test, threshold=threshold)

    report = preparer.get_imbalance_report()

    return {
        "strategy": strategy,
        "target_ratio": target_ratio,
        "model_type": model_type,
        "f1_score": results["f1_score"],
        "precision": results["precision"],
        "recall": results["recall"],
        "accuracy": results["accuracy"],
        "roc_auc": results["roc_auc"],
        "auprc": results["auprc"],
        "confusion_matrix": results["confusion_matrix"],
        "before_ratio": report["before"]["ratio"],
        "after_ratio": report["after"]["ratio"],
        "train_shape": X_train_res.shape,
        "test_shape": X_test.shape,
        "trainer": trainer,
        "preparer": preparer,
    }


def tune_model(X_train, y_train, X_val, y_val, model_type="xgboost", threshold=0.30):
    """Grid search for models optimizing F1 performance strictly at your chosen custom threshold"""
    best_f1 = 0
    best_params = {}

    if model_type == "xgboost":
        for depth in [4, 6, 8]:
            for lr in [0.05, 0.1, 0.2]:
                model = XGBClassifier(
                    max_depth=depth,
                    learning_rate=lr,
                    n_estimators=100,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric="logloss",
                    n_jobs=-1,
                )
                model.fit(X_train, y_train)
                # FIX: Manual probability indexing to enforce the target operational threshold boundary
                y_proba = model.predict_proba(X_val)[:, 1]
                y_pred = (y_proba >= threshold).astype(int)
                f1 = f1_score(y_val, y_pred)
                if f1 > best_f1:
                    best_f1, best_params = f1, {"max_depth": depth, "learning_rate": lr}

    elif model_type == "random_forest":
        for depth in [5, 10, 15]:
            for est in [50, 100, 200]:
                model = RandomForestClassifier(
                    max_depth=depth, n_estimators=est, random_state=42, n_jobs=-1
                )
                model.fit(X_train, y_train)
                # FIX: Applied risk threshold override here as well
                y_proba = model.predict_proba(X_val)[:, 1]
                y_pred = (y_proba >= threshold).astype(int)
                f1 = f1_score(y_val, y_pred)
                if f1 > best_f1:
                    best_f1, best_params = f1, {"max_depth": depth, "n_estimators": est}

    print(
        f"Best {model_type.upper()} params at threshold {threshold}: {best_params} (Validation F1: {best_f1:.4f})"
    )
    return best_params


def compare_models(results_list):
    """Print side-by-side model comparison"""
    print("\n" + "=" * 90)
    print(
        f"{'Model':<15} {'Strategy':<12} {'F1':<8} {'Precision':<10} {'Recall':<8} {'AUPRC':<8}"
    )
    print("-" * 90)
    for r in results_list:
        name = r.get("model_type", "XGBoost").upper()
        print(
            f"{name:<15} {r['strategy']:<12} {r['f1_score']:<8.4f} {r['precision']:<10.4f} {r['recall']:<8.4f} {r['auprc']:<8.4f}"
        )
    print("=" * 90)


def cross_validate_best_config(df, target_col, results_list, cv=5, threshold=0.30):
    """Cross-validates the best configuration securely protecting pipelines from out-of-fold data leakage."""
    best = max(results_list, key=lambda x: x["f1_score"])
    strategy = best["strategy"]
    target_ratio = best["target_ratio"]
    model_type = best["trainer"].model_type

    print(
        f"Rigorous Cross-Validating: Model={model_type.upper()} | Strategy={strategy.upper()} | Ratio={target_ratio} | Threshold={threshold}"
    )

    cols_to_drop = [
        target_col,
        "user_id",
        "device_id",
        "ip_address",
        "signup_time",
        "purchase_time",
        "ip_address_int",
    ]
    X_raw = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    y_raw = df[target_col]

    X_train_raw, _, y_train_raw, _ = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    f1_scores, auprc_scores = [], []

    X_train_df = pd.DataFrame(X_train_raw).reset_index(drop=True)
    y_train_series = pd.Series(y_train_raw).reset_index(drop=True)

    for train_idx, val_idx in skf.split(X_train_df, y_train_series):
        X_tr_raw, X_val_raw = X_train_df.iloc[train_idx].reset_index(
            drop=True
        ), X_train_df.iloc[val_idx].reset_index(drop=True)
        y_tr_fold, y_val_fold = y_train_series.iloc[train_idx].reset_index(
            drop=True
        ), y_train_series.iloc[val_idx].reset_index(drop=True)

        fold_preparer = ModelingPreparer(target_col=target_col)
        cat_cols = X_tr_raw.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        all_nums = X_tr_raw.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [
            col
            for col in all_nums
            if col not in ["is_first_4hours", "is_high_risk_country"]
        ]

        X_tr_fold, X_val_fold = X_tr_raw.copy(), X_val_raw.copy()

        if cat_cols:
            encoded_tr = fold_preparer.encoder.fit_transform(X_tr_raw[cat_cols])
            encoded_val = fold_preparer.encoder.transform(X_val_raw[cat_cols])
            encoded_cols = fold_preparer.encoder.get_feature_names_out(cat_cols)

            df_enc_tr = pd.DataFrame(
                encoded_tr, columns=encoded_cols, index=X_tr_raw.index
            )
            df_enc_val = pd.DataFrame(
                encoded_val, columns=encoded_cols, index=X_val_raw.index
            )
            X_tr_fold = X_tr_fold.drop(columns=cat_cols).join(df_enc_tr)
            X_val_fold = X_val_fold.drop(columns=cat_cols).join(df_enc_val)

        if num_cols:
            X_tr_fold[num_cols] = fold_preparer.scaler.fit_transform(X_tr_raw[num_cols])
            X_val_fold[num_cols] = fold_preparer.scaler.transform(X_val_raw[num_cols])

        if strategy == "none":
            counts = y_tr_fold.value_counts()
            fold_weight = (
                counts.get(0, 1) / counts.get(1, 1) if model_type == "xgboost" else 1
            )
            X_tr_res, y_tr_res = X_tr_fold, y_tr_fold
        else:
            fold_weight = 1
            X_tr_res, y_tr_res = fold_preparer.apply_resampling(
                X_tr_fold, y_tr_fold, strategy=strategy, target_ratio=target_ratio
            )

        fold_trainer = FraudModelTrainer(
            model_type=model_type, scale_pos_weight=fold_weight
        )
        fold_trainer.train(X_tr_res, y_tr_res)

        y_proba = fold_trainer.predict_proba(X_val_fold)
        y_pred = (y_proba >= threshold).astype(int)

        f1_scores.append(f1_score(y_val_fold, y_pred))
        precision_vals, recall_vals, _ = precision_recall_curve(y_val_fold, y_proba)
        auprc_scores.append(auc(recall_vals, precision_vals))

    return {
        "strategy": strategy,
        "target_ratio": target_ratio,
        "model_type": model_type,
        "cv_f1_mean": np.mean(f1_scores),
        "cv_f1_std": np.std(f1_scores),
        "cv_auprc_mean": np.mean(auprc_scores),
        "cv_auprc_std": np.std(auprc_scores),
    }


# ============================================
# VISUALIZATION FUNCTIONS
# ============================================


def plot_strategy_comparison(results_list, dataset_name, save_path_prefix=None):
    """Two side-by-side bar charts - F1 Score and AUPRC"""
    df_plot = pd.DataFrame(
        [
            {"strategy": r["strategy"], "f1_score": r["f1_score"], "auprc": r["auprc"]}
            for r in results_list
        ]
    )

    best_per_strategy = df_plot.loc[df_plot.groupby("strategy")["f1_score"].idxmax()]
    strategies = best_per_strategy["strategy"].tolist()
    f1_scores = best_per_strategy["f1_score"].tolist()
    auprc_scores = best_per_strategy["auprc"].tolist()

    colors = [
        (
            "#2ecc71"
            if s == "smote"
            else (
                "#e74c3c"
                if s == "undersample"
                else "#3498db" if s == "oversample" else "#95a5a6"
            )
        )
        for s in strategies
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    bars1 = ax1.bar(
        strategies, f1_scores, color=colors, edgecolor="black", linewidth=0.5
    )
    ax1.set_ylabel("F1 Score")
    ax1.set_title("F1 Score by Strategy", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars1, f1_scores):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    bars2 = ax2.bar(
        strategies, auprc_scores, color=colors, edgecolor="black", linewidth=0.5
    )
    ax2.set_ylabel("AUPRC")
    ax2.set_title("AUPRC by Strategy", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars2, auprc_scores):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    fig.suptitle(
        f"{dataset_name}: Model Performance Comparison",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    if save_path_prefix:
        plt.savefig(f"{save_path_prefix}_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_ratio_impact(results_list, dataset_name, strategy="smote", save_path=None):
    """Show how different ratios affect performance for a specific strategy"""
    df_plot = pd.DataFrame(
        [
            {
                "target_ratio": r["target_ratio"],
                "f1_score": r["f1_score"],
                "precision": r["precision"],
                "recall": r["recall"],
                "auprc": r["auprc"],
            }
            for r in results_list
            if r["strategy"] == strategy
        ]
    ).dropna(subset=["target_ratio"])

    if len(df_plot) < 2:
        print(
            f"Not enough variation in ratio data to plot trends for {dataset_name} - {strategy}"
        )
        return

    df_plot = df_plot.sort_values("target_ratio")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(
        df_plot["target_ratio"],
        df_plot["f1_score"],
        marker="o",
        linewidth=2,
        markersize=8,
        label="F1 Score",
        color="#2ecc71",
    )
    axes[0].plot(
        df_plot["target_ratio"],
        df_plot["precision"],
        marker="s",
        linewidth=2,
        markersize=8,
        label="Precision",
        color="#3498db",
    )
    axes[0].plot(
        df_plot["target_ratio"],
        df_plot["recall"],
        marker="^",
        linewidth=2,
        markersize=8,
        label="Recall",
        color="#e74c3c",
    )
    axes[0].set_xlabel("Target Ratio")
    axes[0].set_ylabel("Score")
    axes[0].set_title(f"{dataset_name}: Core Metrics vs Ratio")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        df_plot["target_ratio"],
        df_plot["auprc"],
        marker="d",
        linewidth=2,
        markersize=8,
        label="AUPRC",
        color="#9b59b6",
    )
    axes[1].set_xlabel("Target Ratio")
    axes[1].set_ylabel("Score")
    axes[1].set_title(f"{dataset_name}: AUPRC vs Ratio")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix_heatmap(result, dataset_name, save_path=None):
    """Plot confusion matrix heatmap for a single result"""
    cm = result["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
    )

    title = f'{dataset_name}: Confusion Matrix - {result["strategy"].upper()}'
    if result.get("target_ratio"):
        title += f' (ratio={result["target_ratio"]})'
    ax.set_title(title, pad=15)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.text(
        0.5,
        -0.18,
        f'F1: {result["f1_score"]:.4f} | Precision: {result["precision"]:.4f} | Recall: {result["recall"]:.4f}',
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        fontweight="bold",
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_all_confusion_matrices(results_list, dataset_name, save_path=None):
    """Plot confusion matrices for all strategies side by side smoothly without structural index errors"""
    strategies = {}
    for r in results_list:
        strat = r["strategy"]
        if strat not in strategies or r["f1_score"] > strategies[strat]["f1_score"]:
            strategies[strat] = r

    n = len(strategies)
    if n == 0:
        return
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes = np.array([axes]).flatten()

    for idx, (strategy, result) in enumerate(strategies.items()):
        cm = result["confusion_matrix"]
        title = f"{strategy.upper()}"
        if result.get("target_ratio"):
            title += f" (ratio={result['target_ratio']})"

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[idx],
            xticklabels=["Legit", "Fraud"],
            yticklabels=["Legit", "Fraud"],
        )
        axes[idx].set_title(title, fontsize=11, fontweight="bold")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")

    for remaining_ax in axes[n:]:
        remaining_ax.set_visible(False)

    plt.suptitle(f"{dataset_name}: Confusion Matrix Comparison", fontsize=13, y=0.98)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_precision_recall_curve(result, X_test, y_test, dataset_name, save_path=None):
    """Plot Precision-Recall curve dynamically extracting model probability boundaries"""
    y_proba = result["trainer"].predict_proba(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    auprc = auc(recall, precision)

    plt.figure(figsize=(7, 5.5))
    plt.plot(
        recall, precision, linewidth=2, color="darkblue", label=f"AUPRC = {auprc:.4f}"
    )
    plt.fill_between(recall, precision, alpha=0.15, color="darkblue")
    plt.xlabel("Recall (Fraud Detection Rate)")
    plt.ylabel("Precision (Flag Accuracy)")
    plt.title(f'{dataset_name}: Precision-Recall Curve ({result["strategy"].upper()})')
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1.01)
    plt.ylim(0, 1.01)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_roc_curve(result, X_test, y_test, dataset_name, save_path=None):
    """Plots a clean, single ROC curve with name fixed to match imports."""
    y_proba = result["trainer"].predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(
        fpr, tpr, linewidth=2.5, color="#1f77b4", label=f"Model (AUC = {roc_auc:.4f})"
    )
    ax.fill_between(fpr, tpr, alpha=0.10, color="#1f77b4")
    ax.plot(
        [0, 1],
        [0, 1],
        color="#7f7f7f",
        linestyle="--",
        linewidth=1.2,
        label="Random Guess (AUC = 0.5000)",
    )

    ax.plot(
        fpr[optimal_idx],
        tpr[optimal_idx],
        "ro",
        markersize=9,
        label=f"Optimal Geometric Split\nThreshold: {optimal_threshold:.3f}\nFPR: {fpr[optimal_idx]:.3f} | TPR: {tpr[optimal_idx]:.3f}",
    )

    ax.set_xlabel(
        "False Positive Rate (False Alarms)", fontsize=11, fontweight="bold", labelpad=8
    )
    ax.set_ylabel(
        "True Positive Rate (Recall / Detection)",
        fontsize=11,
        fontweight="bold",
        labelpad=8,
    )
    ax.set_title(
        f"{dataset_name}\nROC Performance Summary",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )

    ax.text(
        0.35,
        0.35,
        "Curves closer to the top-left\ncorner represent stronger\npredictive stability.",
        transform=ax.transAxes,
        fontsize=9,
        style="italic",
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor="#fff9db", edgecolor="none", alpha=0.9
        ),
    )

    ax.legend(
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        fontsize=9.5,
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
