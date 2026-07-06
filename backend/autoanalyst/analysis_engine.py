"""
analysis_engine.py
==================
Automated analysis pipeline for the AutoAnalyst module.

Components
----------
1. SmartCleaner  – detects & fixes common data-quality issues.
2. EDAEngine     – generates statistical summaries + visualisation plots.
3. ModelingAdapter – auto-detects problem type and runs a baseline model.
4. AnalysisEngine  – convenience façade that chains all three.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Lazy-import helpers
# ---------------------------------------------------------------------------

def _import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")          # non-interactive backend – safe for servers
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError("Install matplotlib:  pip install matplotlib")

def _import_plotly():
    try:
        import plotly.express as px
        import plotly.io as pio
        return px, pio
    except ImportError:
        raise ImportError("Install plotly:  pip install plotly")

def _import_sklearn():
    try:
        import sklearn
        return sklearn
    except ImportError:
        raise ImportError("Install scikit-learn:  pip install scikit-learn")

def _import_statsmodels():
    try:
        import statsmodels.api as sm
        return sm
    except ImportError:
        raise ImportError("Install statsmodels:  pip install statsmodels")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _df_to_b64_png(fig) -> str:
    """Save a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _numeric_cols(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include="number").columns.tolist()


def _categorical_cols(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def _datetime_cols(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()


# ---------------------------------------------------------------------------
# 1. SmartCleaner
# ---------------------------------------------------------------------------

class SmartCleaner:
    """
    Detects and fixes common data-quality issues in a DataFrame.

    Actions performed
    -----------------
    - Strip leading/trailing whitespace from string columns.
    - Infer better dtypes (int, float, datetime) from object columns.
    - Fill numeric missing values with the column median.
    - Fill categorical missing values with the column mode (or "Unknown").
    - Remove duplicate rows.
    - Remove statistical outliers (IQR method) from numeric columns.
    """

    def __init__(self, remove_outliers: bool = True, iqr_factor: float = 3.0) -> None:
        self.remove_outliers = remove_outliers
        self.iqr_factor = iqr_factor
        self.report: Dict[str, Any] = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean *df* in-place and return it. Populates self.report."""
        df = df.copy()
        self.report = {}

        # 1) Strip strings
        str_cols = df.select_dtypes(include="object").columns
        for col in str_cols:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})

        # 2) Infer dtypes
        df = self._infer_dtypes(df)

        # 3) Report & fix missing values
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        self.report["missing_values"] = missing.to_dict()

        for col in df.columns:
            if df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                else:
                    mode_vals = df[col].mode()
                    fill_val = mode_vals[0] if len(mode_vals) > 0 else "Unknown"
                    df[col] = df[col].fillna(fill_val)

        # 4) Remove duplicates
        n_before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        self.report["duplicates_removed"] = n_before - len(df)

        # 5) Outlier removal
        if self.remove_outliers:
            outlier_rows: pd.Series = pd.Series([False] * len(df), index=df.index)
            num_cols = _numeric_cols(df)
            for col in num_cols:
                q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - self.iqr_factor * iqr
                upper = q3 + self.iqr_factor * iqr
                outlier_rows |= (df[col] < lower) | (df[col] > upper)
            n_outliers = outlier_rows.sum()
            df = df[~outlier_rows].reset_index(drop=True)
            self.report["outliers_removed"] = int(n_outliers)

        self.report["final_shape"] = df.shape
        return df

    def _infer_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Try to convert object columns to numeric or datetime."""
        for col in df.select_dtypes(include="object").columns:
            # Try numeric
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() > 0.8 * df[col].notna().sum():
                df[col] = converted
                continue
            # Try datetime
            converted_dt = pd.to_datetime(df[col], errors="coerce")
            if converted_dt.notna().sum() > 0.8 * df[col].notna().sum():
                df[col] = converted_dt
        return df

    def summary(self) -> str:
        lines = ["**Smart Clean Report**"]
        mv = self.report.get("missing_values", {})
        if mv:
            lines.append(f"- Missing values filled: {mv}")
        lines.append(f"- Duplicate rows removed: {self.report.get('duplicates_removed', 0)}")
        lines.append(f"- Outlier rows removed: {self.report.get('outliers_removed', 0)}")
        lines.append(f"- Final shape: {self.report.get('final_shape', 'n/a')}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. EDAEngine
# ---------------------------------------------------------------------------

class EDAEngine:
    """
    Generates statistical summaries and visualisation plots.

    Returns
    -------
    report  : dict  with keys "stats", "plots"
    Each plot is a base64-encoded PNG string.
    """

    def __init__(self, max_cat_unique: int = 15) -> None:
        self.max_cat_unique = max_cat_unique

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Full EDA pass. Returns report dict."""
        report: Dict[str, Any] = {
            "shape": df.shape,
            "stats": {},
            "plots": {},
        }

        num_cols = _numeric_cols(df)
        cat_cols = _categorical_cols(df)
        dt_cols  = _datetime_cols(df)

        # ------------------------------------------------------------------
        # Statistical summaries
        # ------------------------------------------------------------------
        if num_cols:
            stats_df = df[num_cols].describe().round(3)
            report["stats"]["numeric"] = stats_df.to_dict()

        if cat_cols:
            cat_stats: Dict[str, Any] = {}
            for col in cat_cols:
                vc = df[col].value_counts()
                cat_stats[col] = {
                    "unique_values": int(df[col].nunique()),
                    "top_5": vc.head(5).to_dict(),
                }
            report["stats"]["categorical"] = cat_stats

        # ------------------------------------------------------------------
        # Plots (matplotlib)
        # ------------------------------------------------------------------
        plt = _import_matplotlib()

        # Histograms for numeric columns (max 9)
        if num_cols:
            cols_to_plot = num_cols[:9]
            n = len(cols_to_plot)
            ncols = min(3, n)
            nrows = (n + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
            axes = np.array(axes).flatten() if n > 1 else [axes]
            for i, col in enumerate(cols_to_plot):
                axes[i].hist(df[col].dropna(), bins=30, edgecolor="white", color="#4C72B0")
                axes[i].set_title(f"{col}", fontsize=10)
                axes[i].set_xlabel(col)
                axes[i].set_ylabel("Frequency")
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)
            fig.suptitle("Histograms", fontsize=13, fontweight="bold")
            plt.tight_layout()
            report["plots"]["histograms"] = _df_to_b64_png(fig)
            plt.close(fig)

        # Correlation heatmap
        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            fig, ax = plt.subplots(figsize=(min(12, len(num_cols) + 1), min(10, len(num_cols))))
            import matplotlib.colors as mcolors
            cmap = plt.cm.RdYlGn
            im = ax.imshow(corr.values, aspect="auto", cmap=cmap, vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax)
            ax.set_xticks(range(len(num_cols)))
            ax.set_yticks(range(len(num_cols)))
            ax.set_xticklabels(num_cols, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(num_cols, fontsize=8)
            # Annotate cells
            for r in range(len(num_cols)):
                for c in range(len(num_cols)):
                    ax.text(c, r, f"{corr.values[r, c]:.2f}", ha="center", va="center", fontsize=7)
            ax.set_title("Correlation Heatmap", fontsize=13, fontweight="bold")
            plt.tight_layout()
            report["plots"]["correlation_heatmap"] = _df_to_b64_png(fig)
            plt.close(fig)

        # Scatter plot: top-2 correlated numeric columns
        if len(num_cols) >= 2:
            corr = df[num_cols].corr().abs()
            corr_vals = corr.values.copy()          # writable copy — fixes read-only error
            np.fill_diagonal(corr_vals, 0)
            flat_idx = np.argmax(corr_vals)
            r_idx, c_idx = np.unravel_index(flat_idx, corr_vals.shape)
            col_x, col_y = num_cols[r_idx], num_cols[c_idx]
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(df[col_x], df[col_y], alpha=0.5, s=20, color="#4C72B0")
            ax.set_xlabel(col_x)
            ax.set_ylabel(col_y)
            ax.set_title(f"Scatter: {col_x} vs {col_y}", fontsize=11, fontweight="bold")
            plt.tight_layout()
            report["plots"]["scatter_top_corr"] = _df_to_b64_png(fig)
            plt.close(fig)

        # Bar chart for top categorical column
        if cat_cols:
            col = cat_cols[0]
            if df[col].nunique() <= self.max_cat_unique:
                vc = df[col].value_counts().head(10)
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(vc.index.astype(str), vc.values, color="#DD8452")
                ax.set_xlabel(col)
                ax.set_ylabel("Count")
                ax.set_title(f"Top values — {col}", fontsize=11, fontweight="bold")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                report["plots"]["bar_top_category"] = _df_to_b64_png(fig)
                plt.close(fig)

        # Time-series line chart (if datetime + numeric exist)
        if dt_cols and num_cols:
            dt_col = dt_cols[0]
            val_col = num_cols[0]
            ts = df[[dt_col, val_col]].dropna().sort_values(dt_col)
            if len(ts) > 2:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(ts[dt_col], ts[val_col], linewidth=1.5, color="#55A868")
                ax.set_xlabel("Date")
                ax.set_ylabel(val_col)
                ax.set_title(f"Time Series — {val_col}", fontsize=11, fontweight="bold")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                report["plots"]["time_series"] = _df_to_b64_png(fig)
                plt.close(fig)

        return report


# ---------------------------------------------------------------------------
# 3. ModelingAdapter
# ---------------------------------------------------------------------------

class ModelingAdapter:
    """
    Auto-detects problem type and fits a baseline sklearn model.

    Detection rules
    ---------------
    - If a datetime column exists and there is a numeric target → Time Series (ARIMA).
    - If the target column has ≤ 20 unique values → Classification.
    - Else → Regression.
    """

    def __init__(self, target_col: Optional[str] = None) -> None:
        self.target_col = target_col
        self.model_type: str = "unknown"
        self.metrics: Dict[str, float] = {}
        self.feature_importances: Optional[pd.Series] = None

    def fit(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Auto-fit a baseline model. Returns a result dict with
        'model_type', 'metrics', 'feature_importances', and optionally 'forecast_plot'.
        """
        sklearn = _import_sklearn()
        df = df.copy()
        num_cols = _numeric_cols(df)
        dt_cols  = _datetime_cols(df)

        if not num_cols:
            return {"error": "No numeric columns found — cannot build a model."}

        # Determine target
        target = self.target_col or num_cols[-1]
        if target not in df.columns:
            return {"error": f"Target column '{target}' not found."}

        # Choose problem type
        if dt_cols:
            return self._run_timeseries(df, dt_cols[0], target)

        n_unique_target = df[target].nunique()
        if n_unique_target <= 20 and n_unique_target >= 2:
            self.model_type = "classification"
            return self._run_classification(df, target)
        else:
            self.model_type = "regression"
            return self._run_regression(df, target)

    # ------------------------------------------------------------------

    def _prepare_features(
        self, df: pd.DataFrame, target: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        from sklearn.preprocessing import LabelEncoder

        y = df[target].copy()
        X = df.drop(columns=[target])

        # Drop datetime columns
        X = X.select_dtypes(exclude=["datetime", "datetimetz"])

        # Encode categoricals
        for col in X.select_dtypes(include=["object", "category"]).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

        # Fill any residual NaNs
        X = X.fillna(X.median(numeric_only=True))
        return X, y

    def _run_regression(self, df: pd.DataFrame, target: str) -> Dict[str, Any]:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, r2_score

        X, y = self._prepare_features(df, target)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        self.metrics = {
            "MAE": round(mean_absolute_error(y_test, y_pred), 4),
            "R2":  round(r2_score(y_test, y_pred), 4),
        }
        self.feature_importances = pd.Series(
            model.feature_importances_, index=X.columns
        ).sort_values(ascending=False)

        return {
            "model_type": "regression (GradientBoostingRegressor)",
            "target": target,
            "metrics": self.metrics,
            "feature_importances": self.feature_importances.head(10).to_dict(),
        }

    def _run_classification(self, df: pd.DataFrame, target: str) -> Dict[str, Any]:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report
        from sklearn.preprocessing import LabelEncoder

        X, y = self._prepare_features(df, target)
        le = LabelEncoder()
        y_enc = le.fit_transform(y.astype(str))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.2, random_state=42
        )
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        report_str = classification_report(y_test, y_pred, zero_division=0)
        self.feature_importances = pd.Series(
            model.feature_importances_, index=X.columns
        ).sort_values(ascending=False)

        return {
            "model_type": "classification (RandomForestClassifier)",
            "target": target,
            "classification_report": report_str,
            "feature_importances": self.feature_importances.head(10).to_dict(),
        }

    def _run_timeseries(
        self, df: pd.DataFrame, dt_col: str, target: str
    ) -> Dict[str, Any]:
        sm = _import_statsmodels()
        plt = _import_matplotlib()

        ts = df[[dt_col, target]].dropna().sort_values(dt_col)
        ts = ts.set_index(dt_col)[target]

        try:
            model = sm.tsa.ARIMA(ts, order=(1, 1, 1))
            result = model.fit()
            forecast_steps = min(10, max(1, len(ts) // 5))
            forecast = result.forecast(steps=forecast_steps)

            # Plot
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(ts.index, ts.values, label="Observed", color="#4C72B0")
            ax.plot(forecast.index, forecast.values, label="Forecast", color="#DD8452", linestyle="--")
            ax.set_title(f"ARIMA Forecast — {target}", fontsize=11, fontweight="bold")
            ax.legend()
            plt.tight_layout()
            forecast_b64 = _df_to_b64_png(fig)
            plt.close(fig)

            return {
                "model_type": "time_series (ARIMA(1,1,1))",
                "target": target,
                "forecast": forecast.to_dict(),
                "aic": round(result.aic, 3),
                "forecast_plot": forecast_b64,
            }
        except Exception as exc:
            logger.warning("ARIMA failed: %s — falling back to regression.", exc)
            self.model_type = "regression"
            return self._run_regression(df, target)


# ---------------------------------------------------------------------------
# 4. AnalysisEngine – convenience façade
# ---------------------------------------------------------------------------

class AnalysisEngine:
    """
    Full pipeline: SmartClean → EDA → Modeling.

    Usage
    -----
    engine = AnalysisEngine()
    result = engine.run(df)
    # result.keys(): clean_summary, eda, model
    """

    def __init__(
        self,
        remove_outliers: bool = True,
        target_col: Optional[str] = None,
        skip_modeling: bool = False,
    ) -> None:
        self.remove_outliers = remove_outliers
        self.target_col = target_col
        self.skip_modeling = skip_modeling

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run the full pipeline. Returns a consolidated report dict."""
        # Step 1 – Smart Clean
        cleaner = SmartCleaner(remove_outliers=self.remove_outliers)
        df_clean = cleaner.fit_transform(df)

        # Step 2 – EDA
        eda_engine = EDAEngine()
        eda_report = eda_engine.run(df_clean)

        result: Dict[str, Any] = {
            "clean_summary": cleaner.report,
            "eda": eda_report,
        }

        # Step 3 – Modeling (optional)
        if not self.skip_modeling:
            adapter = ModelingAdapter(target_col=self.target_col)
            model_report = adapter.fit(df_clean)
            result["model"] = model_report

        return result
