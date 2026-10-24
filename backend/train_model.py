"""
backend/train_model.py
======================
Model Training, Validation, and Evaluation Pipeline for AgriPredict AI.

This script implements end-to-end model training for Crop Yield Prediction:
1. Loads cleaned data from 'data/processed/crop_yield_clean.csv'.
2. Performs specialized data filtering:
   - Drops generic crops whose names contain "other" (case-insensitive).
   - Drops minor crops with < 300 rows (e.g. Cashewnut, Arecanut, Black pepper).
   - Drops data entry errors where Yield_t_per_ha < 0.01.
   - Prints the final crop list and counts.
3. Feature schema:
   - Inputs: Crop, Season, Rainfall_mm, Temperature_C, Soil_Type, Area_ha.
   - Target: Yield_t_per_ha.
   - State and Crop_Year are preserved for validation splitting and analysis only.
4. Preprocessing:
   - ColumnTransformer with OneHotEncoder(handle_unknown='ignore') for categoricals
   - StandardScaler for continuous numerical features.
5. Target Transformation:
   - Wraps regressors in TransformedTargetRegressor(func=log1p, inverse_func=expm1)
   - Evaluates performance on original t/ha scale (R², MAE, RMSE) and log scale (R²_log).
6. Dual Validation Strategy:
   - Split A: Time-based split (train on earlier years, test on latest 3 years).
     * NOTE: The time-based split provides the more honest, realistic evaluation for
       agricultural forecasting because production models must generalize forward in
       time rather than interpolate across random samples from the same crop year.
   - Split B: Random 80/20 train/test split (random_state=42).
7. Model Comparison:
   - Naive baseline (historical mean yield per Crop and Season pair)
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor
   - HistGradientBoosting Regressor
   - XGBoost Regressor
   - Comparison table saved to 'models/metrics.json'.
8. Hyperparameter Tuning:
   - Tunes best model with RandomizedSearchCV (cv=3, small n_iter).
9. Per-Crop Test Metrics:
   - Evaluates per-crop R² and MAE on the time-based test set.
   - Saves results to 'models/metrics_per_crop.json'.
   - Prints top 10 best and worst performing crops.
10. Artifact Generation:
   - 'models/best_model.joblib': Full fitted pipeline for inference.
   - 'models/feature_importance.png': Permutation importance on the 6 original input features.
   - 'models/pred_vs_actual.png': Scatter & parity plot of predicted vs actual yield.
   - 'models/metadata.json': Input ranges, valid crop-seasons (>= 50 rows), soil types, and summary.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

# Check XGBoost availability
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Fix environment and random state
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)

# File paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "crop_yield_clean.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

METRICS_PATH = MODELS_DIR / "metrics.json"
METRICS_PER_CROP_PATH = MODELS_DIR / "metrics_per_crop.json"
METADATA_PATH = MODELS_DIR / "metadata.json"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
FEATURE_IMP_PATH = MODELS_DIR / "feature_importance.png"
PRED_ACTUAL_PATH = MODELS_DIR / "pred_vs_actual.png"


# =============================================================================
# Custom Estimator: Naive Crop-Season Mean Baseline
# =============================================================================
class NaiveCropSeasonMeanRegressor(BaseEstimator, RegressorMixin):
    """
    Baseline regressor that predicts the historical mean yield for each
    (Crop, Season) pair. Falls back to Crop mean or global mean if unseen.
    """
    def __init__(self):
        self.crop_season_means_: Dict[Tuple[str, str], float] = {}
        self.crop_means_: Dict[str, float] = {}
        self.global_mean_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series):
        df_temp = X[["Crop", "Season"]].copy()
        df_temp["_target"] = y.values
        self.crop_season_means_ = df_temp.groupby(["Crop", "Season"])["_target"].mean().to_dict()
        self.crop_means_ = df_temp.groupby("Crop")["_target"].mean().to_dict()
        self.global_mean_ = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = []
        for _, row in X.iterrows():
            key = (row["Crop"], row["Season"])
            if key in self.crop_season_means_:
                preds.append(self.crop_season_means_[key])
            elif row["Crop"] in self.crop_means_:
                preds.append(self.crop_means_[row["Crop"]])
            else:
                preds.append(self.global_mean_)
        return np.array(preds)


# =============================================================================
# Step 1-4: Data Loading and Filtering
# =============================================================================
def load_and_filter_data(data_path: Path) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("DATA FILTERING & REFINEMENT")
    print("=" * 70)
    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at '{data_path}'. Please run prepare_data.py first.")

    df = pd.read_csv(data_path)
    initial_rows = len(df)
    print(f"Loaded '{data_path.name}': {initial_rows:,} rows.")

    # 1. Drop generic crops whose name contains 'other' (case-insensitive)
    other_mask = df["Crop"].str.contains("other", case=False, na=False)
    dropped_other_crops = df.loc[other_mask, "Crop"].unique().tolist()
    dropped_other_count = other_mask.sum()
    df = df[~other_mask].copy()
    print(f"1. Generic 'other' crops dropped: {dropped_other_count:,} rows.")
    print(f"   Excluded crop categories: {dropped_other_crops}")

    # 2. Drop crops with < 300 rows after cleaning
    crop_counts = df["Crop"].value_counts()
    low_crops = crop_counts[crop_counts < 300]
    dropped_low_count = low_crops.sum()
    df = df[~df["Crop"].isin(low_crops.index)].copy()
    print(f"2. Low-sample crops (< 300 rows) dropped: {dropped_low_count:,} rows across {len(low_crops)} crops:")
    for crop_name, cnt in low_crops.items():
        print(f"   - '{crop_name}': {cnt} rows")

    # 3. Drop rows with Yield_t_per_ha < 0.01 as likely data errors
    low_yield_mask = df["Yield_t_per_ha"] < 0.01
    dropped_low_yield_count = low_yield_mask.sum()
    df = df[~low_yield_mask].copy()
    print(f"3. Anomalous near-zero yield (Yield < 0.01 t/ha) dropped: {dropped_low_yield_count:,} rows.")

    total_dropped = initial_rows - len(df)
    print(f"\nTotal rows filtered: {total_dropped:,} (Retained: {len(df):,} rows).")

    # 4. Final crop list and row counts
    print(f"\n4. Final Crop List ({df['Crop'].nunique()} major crops retained):")
    final_counts = df["Crop"].value_counts()
    for crop_name, count in final_counts.items():
        print(f"   - {crop_name:<26}: {count:,} rows")

    return df


# =============================================================================
# Step 5 & 8: Data Splitting (Time-Based & Random 80/20)
# =============================================================================
def prepare_splits(df: pd.DataFrame):
    """
    Creates both Time-based split (latest 3 years for test) and Random 80/20 split.
    
    ACADEMIC NOTE ON TIME-BASED VALIDATION:
    ---------------------------------------
    The time-based split is the more honest and representative evaluation metric
    for agricultural decision-support systems. In practical deployment, models are
    trained on historical records and evaluated on upcoming, unseen future crop years.
    Random splits suffer from optimistic leakage because weather and macroeconomic
    conditions from the same calendar year appear across both train and test partitions.
    """
    feature_cols = ["Crop", "Season", "Rainfall_mm", "Temperature_C", "Soil_Type", "Area_ha"]
    target_col = "Yield_t_per_ha"

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Time-based split: latest 3 years in test
    all_years = sorted(df["Crop_Year"].unique())
    latest_3_years = all_years[-3:]
    time_train_mask = ~df["Crop_Year"].isin(latest_3_years)
    time_test_mask = df["Crop_Year"].isin(latest_3_years)

    X_train_time = X[time_train_mask].copy()
    y_train_time = y[time_train_mask].copy()
    X_test_time = X[time_test_mask].copy()
    y_test_time = y[time_test_mask].copy()

    print("\n" + "=" * 70)
    print("VALIDATION SPLITS CONFIGURATION")
    print("=" * 70)
    print(f"Time-Based Split:")
    print(f"  - Training years: {all_years[0]} - {all_years[-4]} ({len(X_train_time):,} samples, {len(X_train_time)/len(df)*100:.1f}%)")
    print(f"  - Testing years : {latest_3_years} ({len(X_test_time):,} samples, {len(X_test_time)/len(df)*100:.1f}%)")

    # Random 80/20 split
    X_train_rand, X_test_rand, y_train_rand, y_test_rand = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    print(f"Random 80/20 Split (random_state={RANDOM_STATE}):")
    print(f"  - Training samples: {len(X_train_rand):,} (80.0%)")
    print(f"  - Testing samples : {len(X_test_rand):,} (20.0%)")

    return {
        "time": (X_train_time, X_test_time, y_train_time, y_test_time, latest_3_years),
        "random": (X_train_rand, X_test_rand, y_train_rand, y_test_rand),
        "full_X": X,
        "full_y": y,
    }


# =============================================================================
# Step 6 & 7: Pipeline Builder with Preprocessing & TransformedTargetRegressor
# =============================================================================
def build_pipeline(regressor_estimator) -> TransformedTargetRegressor:
    """
    Constructs ColumnTransformer and wraps the resulting pipeline in
    TransformedTargetRegressor with log1p / expm1 to handle yield skewness.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["Rainfall_mm", "Temperature_C", "Area_ha"]),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["Crop", "Season", "Soil_Type"],
            ),
        ]
    )

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", regressor_estimator),
    ])

    return TransformedTargetRegressor(
        regressor=pipe,
        func=np.log1p,
        inverse_func=np.expm1,
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes R², MAE, RMSE in original t/ha scale and R² in log scale."""
    y_pred_clipped = np.maximum(y_pred, 0)
    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    r2_log = float(r2_score(np.log1p(y_true), np.log1p(y_pred_clipped)))

    return {
        "R2": round(r2, 4),
        "R2_log": round(r2_log, 4),
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
    }


# =============================================================================
# Step 9: Models Comparison
# =============================================================================
def compare_models(splits: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any], Any]:
    print("\n" + "=" * 70)
    print("STEP 9: BENCHMARKING MODELS UNDER DUAL VALIDATION SPLITS")
    print("=" * 70)

    X_tr_t, X_te_t, y_tr_t, y_te_t, _ = splits["time"]
    X_tr_r, X_te_r, y_tr_r, y_te_r = splits["random"]

    # Define candidates
    model_factories = {
        "Naive Mean Baseline": lambda: NaiveCropSeasonMeanRegressor(),
        "Linear Regression": lambda: LinearRegression(),
        "Decision Tree": lambda: DecisionTreeRegressor(max_depth=12, random_state=RANDOM_STATE),
        "Random Forest": lambda: RandomForestRegressor(
            n_estimators=100, max_depth=15, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "HistGradientBoosting": lambda: HistGradientBoostingRegressor(
            max_iter=100, random_state=RANDOM_STATE
        ),
    }

    if XGBOOST_AVAILABLE:
        model_factories["XGBoost"] = lambda: xgb.XGBRegressor(
            n_estimators=120,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    else:
        print("Note: XGBoost is not installed; skipping gracefully.")

    metrics_records = []
    trained_time_models = {}

    for name, factory in model_factories.items():
        print(f"\nTraining & Evaluating '{name}'...")
        t0 = time.time()

        if name == "Naive Mean Baseline":
            # Naive baseline handles its own fitting on X_tr_t and X_tr_r directly
            model_time = factory().fit(X_tr_t, y_tr_t)
            preds_time = model_time.predict(X_te_t)
            time_metrics = compute_metrics(y_te_t.values, preds_time)

            model_rand = factory().fit(X_tr_r, y_tr_r)
            preds_rand = model_rand.predict(X_te_r)
            rand_metrics = compute_metrics(y_te_r.values, preds_rand)
            trained_time_models[name] = model_time
        else:
            # Time-based split evaluation
            model_time = build_pipeline(factory())
            model_time.fit(X_tr_t, y_tr_t)
            preds_time = model_time.predict(X_te_t)
            time_metrics = compute_metrics(y_te_t.values, preds_time)
            trained_time_models[name] = model_time

            # Random split evaluation
            model_rand = build_pipeline(factory())
            model_rand.fit(X_tr_r, y_tr_r)
            preds_rand = model_rand.predict(X_te_r)
            rand_metrics = compute_metrics(y_te_r.values, preds_rand)

        duration = time.time() - t0
        print(f"  Time-Based Split -> R²={time_metrics['R2']:.3f} | R²_log={time_metrics['R2_log']:.3f} | MAE={time_metrics['MAE']:.3f} t/ha | RMSE={time_metrics['RMSE']:.3f} t/ha")
        print(f"  Random 80/20 Split-> R²={rand_metrics['R2']:.3f} | R²_log={rand_metrics['R2_log']:.3f} | MAE={rand_metrics['MAE']:.3f} t/ha | RMSE={rand_metrics['RMSE']:.3f} t/ha")
        print(f"  Completed in {duration:.1f}s")

        metrics_records.append({
            "model": name,
            "time_based_r2": time_metrics["R2"],
            "time_based_r2_log": time_metrics["R2_log"],
            "time_based_mae": time_metrics["MAE"],
            "time_based_rmse": time_metrics["RMSE"],
            "random_split_r2": rand_metrics["R2"],
            "random_split_r2_log": rand_metrics["R2_log"],
            "random_split_mae": rand_metrics["MAE"],
            "random_split_rmse": rand_metrics["RMSE"],
        })

    metrics_df = pd.DataFrame(metrics_records)

    # Save to metrics.json
    metrics_json_data = {
        "evaluation_notes": {
            "time_based_split": "Trained on 1997-2012, tested on 2013-2015. More honest metric for prospective yield forecasting.",
            "random_split": "Random 80/20 train/test split. Exhibits higher scores due to temporal shared variance.",
        },
        "models": metrics_records,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_json_data, f, indent=2)
    print(f"\nModel metrics saved to '{METRICS_PATH.name}'.")

    # Print comparison table
    print("\n" + "=" * 90)
    print(f"{'MODEL COMPARISON TABLE':^90}")
    print("=" * 90)
    header = f"{'Model':<22} | {'Time R²':<8} {'Time MAE':<9} {'Time RMSE':<10} | {'Rand R²':<8} {'Rand MAE':<9} {'Rand RMSE':<10}"
    print(header)
    print("-" * 90)
    for r in metrics_records:
        row_str = (
            f"{r['model']:<22} | "
            f"{r['time_based_r2']:<8.3f} {r['time_based_mae']:<9.3f} {r['time_based_rmse']:<10.3f} | "
            f"{r['random_split_r2']:<8.3f} {r['random_split_mae']:<9.3f} {r['random_split_rmse']:<10.3f}"
        )
        print(row_str)
    print("=" * 90)

    # Identify best performing model based on time-based R²
    best_row = metrics_df.sort_values(by="time_based_r2", ascending=False).iloc[0]
    best_name = best_row["model"]
    print(f"\nSelected Best Performing Model Architecture: '{best_name}'")

    return metrics_df, metrics_json_data, trained_time_models


# =============================================================================
# Step 10: Hyperparameter Tuning of the Best Model
# =============================================================================
def tune_best_model(splits: Dict[str, Any], best_name: str) -> Tuple[Any, Dict[str, Any]]:
    print("\n" + "=" * 70)
    print("STEP 10: HYPERPARAMETER TUNING (RandomizedSearchCV, cv=3)")
    print("=" * 70)

    X_tr_t, X_te_t, y_tr_t, y_te_t, _ = splits["time"]
    X_tr_r, X_te_r, y_tr_r, y_te_r = splits["random"]

    # Subsample training data if dataset is huge for fast cross-validation on laptop
    tune_sample_size = min(45000, len(X_tr_t))
    tune_indices = np.random.choice(len(X_tr_t), size=tune_sample_size, replace=False)
    X_tune = X_tr_t.iloc[tune_indices]
    y_tune = y_tr_t.iloc[tune_indices]
    print(f"Tuning search running on representative subsample of {tune_sample_size:,} records (cv=3)...")

    if best_name == "XGBoost" and XGBOOST_AVAILABLE:
        base_estimator = xgb.XGBRegressor(n_jobs=-1, random_state=RANDOM_STATE)
        param_dist = {
            "regressor__regressor__n_estimators": [100, 150, 200],
            "regressor__regressor__max_depth": [5, 6, 7],
            "regressor__regressor__learning_rate": [0.05, 0.08, 0.12],
            "regressor__regressor__subsample": [0.8, 0.9, 1.0],
            "regressor__regressor__colsample_bytree": [0.8, 0.9, 1.0],
        }
    else:
        # Default fallback to HistGradientBoosting or Random Forest
        base_estimator = HistGradientBoostingRegressor(random_state=RANDOM_STATE)
        param_dist = {
            "regressor__regressor__max_iter": [100, 150, 200],
            "regressor__regressor__max_depth": [8, 12, 16, None],
            "regressor__regressor__learning_rate": [0.04, 0.08, 0.12],
            "regressor__regressor__min_samples_leaf": [20, 40, 60],
        }

    pipeline_to_tune = build_pipeline(base_estimator)

    search = RandomizedSearchCV(
        estimator=pipeline_to_tune,
        param_distributions=param_dist,
        n_iter=6,
        cv=3,
        scoring="r2",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    t0 = time.time()
    search.fit(X_tune, y_tune)
    tune_time = time.time() - t0
    print(f"RandomizedSearchCV completed in {tune_time:.1f}s.")
    print(f"Best CV R² Score (Log Space): {search.best_score_:.4f}")
    print(f"Best Hyperparameters: {search.best_params_}")

    # Refit best estimator on the FULL time-based training set
    print("Refitting best tuned pipeline on the complete historical training set (138k rows)...")
    best_estimator = search.best_estimator_
    best_estimator.fit(X_tr_t, y_tr_t)

    # Evaluate tuned model on time-based test set
    preds_time = best_estimator.predict(X_te_t)
    time_metrics = compute_metrics(y_te_t.values, preds_time)
    print(f"\nTuned Model Performance (Time-Based Test Set):")
    print(f"  • R²     : {time_metrics['R2']:.4f}")
    print(f"  • R²_log : {time_metrics['R2_log']:.4f}")
    print(f"  • MAE    : {time_metrics['MAE']:.4f} t/ha")
    print(f"  • RMSE   : {time_metrics['RMSE']:.4f} t/ha")

    # Evaluate on random test set as well
    preds_rand = best_estimator.predict(X_te_r)
    rand_metrics = compute_metrics(y_te_r.values, preds_rand)

    tuning_summary = {
        "best_params": search.best_params_,
        "best_cv_score": round(float(search.best_score_), 4),
        "time_based_metrics": time_metrics,
        "random_split_metrics": rand_metrics,
    }

    return best_estimator, tuning_summary


# =============================================================================
# Step 11: Per-Crop Test Metrics (Time-Based Split)
# =============================================================================
def evaluate_per_crop(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
    print("\n" + "=" * 70)
    print("STEP 11: PER-CROP EVALUATION (Time-Based Test Set)")
    print("=" * 70)

    preds = model.predict(X_test)
    eval_df = X_test[["Crop"]].copy()
    eval_df["y_true"] = y_test.values
    eval_df["y_pred"] = preds

    per_crop_metrics = {}
    for crop_name, group in eval_df.groupby("Crop"):
        y_c_true = group["y_true"].values
        y_c_pred = group["y_pred"].values
        n_samples = len(group)

        # Standard R² (handle edge-case of zero variance)
        if len(np.unique(y_c_true)) > 1:
            c_r2 = float(r2_score(y_c_true, y_c_pred))
        else:
            c_r2 = 0.0

        c_mae = float(mean_absolute_error(y_c_true, y_c_pred))
        c_rmse = float(root_mean_squared_error(y_c_true, y_c_pred))
        avg_yield = float(np.mean(y_c_true))

        per_crop_metrics[crop_name] = {
            "samples": int(n_samples),
            "R2": round(c_r2, 4),
            "MAE": round(c_mae, 4),
            "RMSE": round(c_rmse, 4),
            "mean_yield_t_ha": round(avg_yield, 4),
        }

    # Save to models/metrics_per_crop.json
    with open(METRICS_PER_CROP_PATH, "w") as f:
        json.dump(per_crop_metrics, f, indent=2)
    print(f"Per-crop metrics saved to '{METRICS_PER_CROP_PATH.name}'.")

    # Sort crops by R2 descending
    sorted_by_r2 = sorted(per_crop_metrics.items(), key=lambda item: item[1]["R2"], reverse=True)

    print("\nTop 10 Best Performing Crops (by Time-Based R²):")
    print(f"{'Crop':<24} | {'Test Samples':<13} | {'R²':<7} | {'MAE (t/ha)':<11} | {'Mean Yield':<10}")
    print("-" * 72)
    for crop_name, m in sorted_by_r2[:10]:
        print(f"{crop_name:<24} | {m['samples']:<13} | {m['R2']:<7.3f} | {m['MAE']:<11.3f} | {m['mean_yield_t_ha']:<10.2f}")

    print("\nTop 10 Worst Performing Crops (by Time-Based R²):")
    print(f"{'Crop':<24} | {'Test Samples':<13} | {'R²':<7} | {'MAE (t/ha)':<11} | {'Mean Yield':<10}")
    print("-" * 72)
    for crop_name, m in sorted_by_r2[-10:]:
        print(f"{crop_name:<24} | {m['samples']:<13} | {m['R2']:<7.3f} | {m['MAE']:<11.3f} | {m['mean_yield_t_ha']:<10.2f}")

    return per_crop_metrics


# =============================================================================
# Step 12: Visualizations (Permutation Importance & Pred vs Actual)
# =============================================================================
def generate_visualizations(model: Any, X_test: pd.DataFrame, y_test: pd.Series):
    print("\n" + "=" * 70)
    print("STEP 12: GENERATING VISUALIZATIONS (Permutation Importance & Pred vs Actual)")
    print("=" * 70)

    # 1. Permutation Importance on the 6 original input features
    print("Calculating permutation importance on the original 6 input features...")
    # Subsample test set to 4,000 for fast, robust permutation importance
    eval_sample_size = min(4000, len(X_test))
    eval_indices = np.random.choice(len(X_test), size=eval_sample_size, replace=False)
    X_perm = X_test.iloc[eval_indices]
    y_perm = y_test.iloc[eval_indices]

    perm = permutation_importance(
        model,
        X_perm,
        y_perm,
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    feature_names = list(X_test.columns)
    sorted_idx = np.argsort(perm.importances_mean)

    plt.figure(figsize=(9, 5), dpi=300)
    plt.barh(
        range(len(sorted_idx)),
        perm.importances_mean[sorted_idx],
        xerr=perm.importances_std[sorted_idx],
        align="center",
        color="#2b7a4b",
        edgecolor="#184b2c",
        alpha=0.88,
        capsize=4,
    )
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx], fontsize=11, fontweight="medium")
    plt.xlabel("Mean Drop in R² (Permutation Importance)", fontsize=11, fontweight="bold")
    plt.title("Feature Importance on Original Input Columns (AgriPredict AI)", fontsize=13, fontweight="bold", pad=12)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FEATURE_IMP_PATH, dpi=300)
    plt.close()
    print(f"Saved feature importance chart to '{FEATURE_IMP_PATH.name}'.")

    # 2. Predicted vs Actual Yield Plot
    print("Generating Predicted vs Actual Yield visualization...")
    preds = model.predict(X_test)
    r2_val = r2_score(y_test, preds)
    mae_val = mean_absolute_error(y_test, preds)
    rmse_val = root_mean_squared_error(y_test, preds)

    plt.figure(figsize=(8, 7), dpi=300)
    # Scatter sample for crisp rendering without overplotting
    plot_sample_size = min(5000, len(y_test))
    plot_idx = np.random.choice(len(y_test), size=plot_sample_size, replace=False)
    y_true_sample = y_test.iloc[plot_idx]
    y_pred_sample = preds[plot_idx]

    plt.scatter(
        y_true_sample,
        y_pred_sample,
        alpha=0.35,
        color="#1f5f38",
        edgecolors="none",
        s=22,
        label="Test Observations (2013-2015)",
    )

    # Reference parity line (y = x)
    max_limit = min(float(np.percentile(y_test, 99.5)), 60.0)
    plt.plot([0, max_limit], [0, max_limit], color="#d9534f", linestyle="--", linewidth=2.0, label="Perfect Agreement (y = x)")

    plt.xlim(0, max_limit)
    plt.ylim(0, max_limit)
    plt.xlabel("Actual Crop Yield (tonnes / hectare)", fontsize=11, fontweight="bold")
    plt.ylabel("Predicted Crop Yield (tonnes / hectare)", fontsize=11, fontweight="bold")
    plt.title("Predicted vs. Actual Crop Yield (Time-Based Test Set)", fontsize=13, fontweight="bold", pad=12)

    # Metrics annotation box
    metrics_text = f"Test R²  = {r2_val:.3f}\nMAE    = {mae_val:.2f} t/ha\nRMSE   = {rmse_val:.2f} t/ha"
    plt.gca().text(
        0.05,
        0.92,
        metrics_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.9),
    )

    plt.legend(loc="lower right", framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.tight_layout()
    plt.savefig(PRED_ACTUAL_PATH, dpi=300)
    plt.close()
    print(f"Saved predicted vs actual chart to '{PRED_ACTUAL_PATH.name}'.")


# =============================================================================
# Step 13: Metadata Generation
# =============================================================================
def generate_metadata(
    df: pd.DataFrame,
    best_name: str,
    tuning_summary: Dict[str, Any],
):
    print("\n" + "=" * 70)
    print("STEP 13: GENERATING METADATA FOR INFERENCE & API INTEGRATION")
    print("=" * 70)

    # Valid categoricals
    valid_crops = sorted(df["Crop"].unique().tolist())
    valid_seasons = sorted(df["Season"].unique().tolist())
    valid_soil_types = sorted(df["Soil_Type"].unique().tolist())

    # Valid crop-season combinations (having at least 50 rows in the data)
    crop_season_counts = df.groupby(["Crop", "Season"]).size()
    valid_crop_seasons = {}
    average_yield_per_crop_season = {}

    for (crop, season), count in crop_season_counts.items():
        if count >= 50:
            valid_crop_seasons.setdefault(crop, []).append(season)
            avg_y = float(df[(df["Crop"] == crop) & (df["Season"] == season)]["Yield_t_per_ha"].mean())
            average_yield_per_crop_season[f"{crop}__{season}"] = round(avg_y, 4)

    # Input feature ranges (1st and 99th percentiles, min, max, median, mean)
    numerical_features = ["Rainfall_mm", "Temperature_C", "Area_ha"]
    input_ranges = {}
    for col in numerical_features:
        input_ranges[col] = {
            "p01": round(float(df[col].quantile(0.01)), 2),
            "p99": round(float(df[col].quantile(0.99)), 2),
            "min": round(float(df[col].min()), 2),
            "max": round(float(df[col].max()), 2),
            "mean": round(float(df[col].mean()), 2),
            "median": round(float(df[col].median()), 2),
        }

    metadata = {
        "project": "AgriPredict AI",
        "description": "Metadata specification for frontend form validation, parameter ranges, and API model serving.",
        "valid_crops": valid_crops,
        "valid_seasons": valid_seasons,
        "valid_soil_types": valid_soil_types,
        "valid_crop_seasons": valid_crop_seasons,
        "input_ranges": input_ranges,
        "average_yield_per_crop_season": average_yield_per_crop_season,
        "best_model": {
            "architecture": best_name,
            "hyperparameters": tuning_summary["best_params"],
            "cv_score_r2_log": tuning_summary["best_cv_score"],
            "time_based_metrics": tuning_summary["time_based_metrics"],
            "random_split_metrics": tuning_summary["random_split_metrics"],
        },
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved successfully to '{METADATA_PATH.name}'.")
    return metadata


# =============================================================================
# Main Pipeline Runner
# =============================================================================
def main():
    total_start = time.time()
    print("=" * 70)
    print("AgriPredict AI: Crop Yield Model Training Pipeline (Stage 2)")
    print(f"Working Directory: {PROJECT_ROOT}")
    print(f"Random State: {RANDOM_STATE}")
    print("=" * 70)

    # 1-4. Data filtering
    df = load_and_filter_data(DATA_PATH)

    # 5 & 8. Prepare validation splits
    splits = prepare_splits(df)

    # 6, 7 & 9. Model comparison
    metrics_df, metrics_json, trained_models = compare_models(splits)

    # 10. Tune best model
    best_name = metrics_df.sort_values(by="time_based_r2", ascending=False).iloc[0]["model"]
    best_estimator, tuning_summary = tune_best_model(splits, best_name)

    # 11. Per-crop evaluation
    X_tr_t, X_te_t, y_tr_t, y_te_t, _ = splits["time"]
    per_crop_metrics = evaluate_per_crop(best_estimator, X_te_t, y_te_t)

    # 12. Visualizations
    generate_visualizations(best_estimator, X_te_t, y_te_t)

    # 13. Metadata
    generate_metadata(df, best_name, tuning_summary)

    # Save fitted model pipeline
    joblib.dump(best_estimator, BEST_MODEL_PATH)
    print(f"\nSuccessfully serialized production model to '{BEST_MODEL_PATH.name}'.")

    total_time = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"STAGE 2 MODEL TRAINING PIPELINE COMPLETED IN {total_time:.1f} SECONDS")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
