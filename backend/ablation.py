"""
backend/ablation.py
===================
Ablation Study on Feature Configurations and Naive Baseline.

Evaluates four feature & baseline configurations under the identical time-based
split (train 1997-2012, test 2013-2015) using the tuned HistGradientBoosting
hyperparameters and TransformedTargetRegressor(func=log1p, inverse_func=expm1)
established in Stage 2.

Variants:
  A) Current features: Crop, Season, Rainfall_mm, Temperature_C, Soil_Type, Area_ha
  B) A without Area_ha: Crop, Season, Rainfall_mm, Temperature_C, Soil_Type
  C) B with State added as categorical and Soil_Type removed: Crop, Season, Rainfall_mm, Temperature_C, State
  D) Naive Crop x Season historical mean baseline

Metrics Reported:
  - Time-based R²
  - Time-based R²_log
  - Time-based MAE (t/ha)
  - Time-based RMSE (t/ha)
  - Macro-average over crops of (per-crop MAE / that crop's mean yield) as a percentage (%)

Artifacts Produced:
  - models/ablation.json
  - models/variants/variant_A.joblib
  - models/variants/variant_B.joblib
  - models/variants/variant_C.joblib
  - models/variants/variant_D.joblib

NOTE: Preserves models/best_model.joblib and models/metadata.json intact.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Fix environment and random state early to suppress loky core lookup warnings
RANDOM_STATE = 42
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Support serialization and deserialization whether executed directly or imported
if __name__ == "__main__":
    sys.modules["backend.ablation"] = sys.modules["__main__"]

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

np.random.seed(RANDOM_STATE)

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "crop_yield_clean.csv"
MODELS_DIR = PROJECT_ROOT / "models"
VARIANTS_DIR = MODELS_DIR / "variants"
ABLATION_JSON_PATH = MODELS_DIR / "ablation.json"

# Fixed tuned HistGradientBoosting hyperparameters from Stage 2 (train_model.py)
TUNED_HGB_PARAMS = {
    "max_iter": 150,
    "max_depth": None,
    "learning_rate": 0.08,
    "min_samples_leaf": 40,
    "random_state": RANDOM_STATE,
}


# =============================================================================
# Custom Estimator: Naive Crop x Season Mean Baseline (Variant D)
# =============================================================================
class NaiveCropSeasonMeanRegressor(BaseEstimator, RegressorMixin):
    """
    Baseline regressor that predicts the historical training mean yield for each
    (Crop, Season) pair. Falls back to Crop mean or global mean if unseen.
    """
    def __init__(self):
        self.crop_season_means_: Dict[Tuple[str, str], float] = {}
        self.crop_means_: Dict[str, float] = {}
        self.global_mean_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series):
        crops = X["Crop"].values if isinstance(X, pd.DataFrame) else X[:, 0]
        seasons = X["Season"].values if isinstance(X, pd.DataFrame) else X[:, 1]
        df_temp = pd.DataFrame({
            "Crop": crops,
            "Season": seasons,
            "_target": np.asarray(y, dtype=float),
        })
        self.crop_season_means_ = df_temp.groupby(["Crop", "Season"])["_target"].mean().to_dict()
        self.crop_means_ = df_temp.groupby("Crop")["_target"].mean().to_dict()
        self.global_mean_ = float(np.mean(y))
        return self

    def predict(self, X: Any) -> np.ndarray:
        crops = X["Crop"].values if isinstance(X, pd.DataFrame) else X[:, 0]
        seasons = X["Season"].values if isinstance(X, pd.DataFrame) else X[:, 1]
        preds = [
            self.crop_season_means_.get((c, s), self.crop_means_.get(c, self.global_mean_))
            for c, s in zip(crops, seasons)
        ]
        return np.array(preds, dtype=float)


# =============================================================================
# Data Loading and Filtering (Identical to train_model.py)
# =============================================================================
def load_and_filter_data(data_path: Path) -> pd.DataFrame:
    """
    Loads and filters dataset using identical business rules to train_model.py:
    1. Drop generic crops with 'other' in the name.
    2. Drop minor crops with < 300 rows.
    3. Drop anomalous near-zero yields (< 0.01 t/ha).
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at '{data_path}'.")

    df = pd.read_csv(data_path)
    initial_rows = len(df)

    # 1. Drop generic crops whose name contains 'other' (case-insensitive)
    other_mask = df["Crop"].str.contains("other", case=False, na=False)
    df = df[~other_mask].copy()

    # 2. Drop crops with < 300 rows after cleaning
    crop_counts = df["Crop"].value_counts()
    low_crops = crop_counts[crop_counts < 300]
    df = df[~df["Crop"].isin(low_crops.index)].copy()

    # 3. Drop rows with Yield_t_per_ha < 0.01 as likely entry errors
    low_yield_mask = df["Yield_t_per_ha"] < 0.01
    df = df[~low_yield_mask].copy()

    print(f"Data Filtering Complete: {initial_rows:,} -> {len(df):,} rows retained across {df['Crop'].nunique()} crops.")
    return df


# =============================================================================
# Time-Based Split (Identical to train_model.py)
# =============================================================================
def get_time_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits into:
      - Train set: 1997-2012 (all historical years except latest 3)
      - Test set : 2013-2015 (latest 3 years)
    """
    all_years = sorted(df["Crop_Year"].unique())
    latest_3_years = all_years[-3:]
    train_mask = ~df["Crop_Year"].isin(latest_3_years)
    test_mask = df["Crop_Year"].isin(latest_3_years)

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    print(f"Time-Based Split Configuration:")
    print(f"  - Train set ({all_years[0]}-{all_years[-4]}): {len(train_df):,} rows")
    print(f"  - Test set  ({latest_3_years}): {len(test_df):,} rows across {test_df['Crop'].nunique()} crops")
    return train_df, test_df


# =============================================================================
# Pipeline Builders
# =============================================================================
def build_variant_pipeline(num_cols: List[str], cat_cols: List[str]) -> TransformedTargetRegressor:
    """
    Constructs a ColumnTransformer + HistGradientBoosting Pipeline
    wrapped in TransformedTargetRegressor(func=log1p, inverse_func=expm1).
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ]
    )

    base_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", HistGradientBoostingRegressor(**TUNED_HGB_PARAMS)),
    ])

    return TransformedTargetRegressor(
        regressor=base_pipe,
        func=np.log1p,
        inverse_func=np.expm1,
    )


# =============================================================================
# Metrics Evaluation
# =============================================================================
def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    crop_series: pd.Series,
) -> Dict[str, Any]:
    """
    Computes:
      - R² (original scale)
      - R²_log (log1p scale with negative clip safeguard)
      - MAE (t/ha)
      - RMSE (t/ha)
      - Macro-average over crops of (per-crop MAE / that crop's mean yield) as percentage (%)
      - Per-crop detail breakdown
    """
    y_pred_clipped = np.maximum(y_pred, 0)
    r2 = float(r2_score(y_true, y_pred))
    r2_log = float(r2_score(np.log1p(y_true), np.log1p(y_pred_clipped)))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))

    # Per-crop relative MAE
    eval_df = pd.DataFrame({
        "Crop": crop_series.values,
        "y_true": y_true,
        "y_pred": y_pred,
    })

    per_crop_dict = {}
    rel_maes = []

    for crop_name, grp in eval_df.groupby("Crop"):
        c_true = grp["y_true"].values
        c_pred = grp["y_pred"].values
        c_mae = float(mean_absolute_error(c_true, c_pred))
        c_mean = float(np.mean(c_true))
        c_rel = (c_mae / c_mean) if c_mean > 0 else 0.0

        rel_maes.append(c_rel)
        per_crop_dict[crop_name] = {
            "samples": int(len(grp)),
            "crop_mae": round(c_mae, 4),
            "crop_mean_yield": round(c_mean, 4),
            "relative_mae_pct": round(c_rel * 100, 2),
        }

    macro_avg_pct = float(np.mean(rel_maes) * 100)

    return {
        "r2": round(r2, 4),
        "r2_log": round(r2_log, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "macro_crop_rel_mae_pct": round(macro_avg_pct, 2),
        "per_crop_breakdown": per_crop_dict,
    }


# =============================================================================
# Main Ablation Runner
# =============================================================================
def run_ablation():
    print("=" * 80)
    print("AgriPredict AI: Model Feature Ablation Study")
    print(f"Time-Based Split: Train 1997-2012 | Test 2013-2015")
    print(f"HistGradientBoosting Parameters: {TUNED_HGB_PARAMS}")
    print("=" * 80)

    VARIANTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load and prepare split
    df = load_and_filter_data(DATA_PATH)
    train_df, test_df = get_time_split(df)

    y_train = train_df["Yield_t_per_ha"]
    y_test = test_df["Yield_t_per_ha"].values
    crop_test = test_df["Crop"]

    # 2. Define variants
    variants_spec = [
        {
            "id": "A",
            "name": "Current features",
            "description": "Crop, Season, Rainfall_mm, Temperature_C, Soil_Type, Area_ha",
            "features": ["Crop", "Season", "Rainfall_mm", "Temperature_C", "Soil_Type", "Area_ha"],
            "type": "model",
            "num_cols": ["Rainfall_mm", "Temperature_C", "Area_ha"],
            "cat_cols": ["Crop", "Season", "Soil_Type"],
        },
        {
            "id": "B",
            "name": "A without Area_ha",
            "description": "Crop, Season, Rainfall_mm, Temperature_C, Soil_Type",
            "features": ["Crop", "Season", "Rainfall_mm", "Temperature_C", "Soil_Type"],
            "type": "model",
            "num_cols": ["Rainfall_mm", "Temperature_C"],
            "cat_cols": ["Crop", "Season", "Soil_Type"],
        },
        {
            "id": "C",
            "name": "B with State (Soil_Type removed)",
            "description": "Crop, Season, Rainfall_mm, Temperature_C, State",
            "features": ["Crop", "Season", "Rainfall_mm", "Temperature_C", "State"],
            "type": "model",
            "num_cols": ["Rainfall_mm", "Temperature_C"],
            "cat_cols": ["Crop", "Season", "State"],
        },
        {
            "id": "D",
            "name": "Crop x Season mean baseline",
            "description": "Historical training mean yield per (Crop, Season) pair",
            "features": ["Crop", "Season"],
            "type": "baseline",
        },
    ]

    results = []

    for spec in variants_spec:
        v_id = spec["id"]
        v_name = spec["name"]
        print(f"\nTraining Variant {v_id}: {v_name}...")
        t0 = time.time()

        if spec["type"] == "model":
            pipeline = build_variant_pipeline(spec["num_cols"], spec["cat_cols"])
            X_train = train_df[spec["features"]]
            X_test = test_df[spec["features"]]
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
        else:
            # Variant D: Naive Crop x Season mean baseline inside Pipeline
            pipeline = Pipeline([
                ("naive_baseline", NaiveCropSeasonMeanRegressor())
            ])
            X_train = train_df[spec["features"]]
            X_test = test_df[spec["features"]]
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)

        train_time = time.time() - t0
        metrics = evaluate_predictions(y_test, preds, crop_test)

        # Save pipeline to models/variants/
        variant_filename = f"variant_{v_id}.joblib"
        variant_filepath = VARIANTS_DIR / variant_filename
        joblib.dump(pipeline, variant_filepath)
        print(f"  Saved pipeline to: {variant_filepath.relative_to(PROJECT_ROOT)}")
        print(f"  Completed in {train_time:.1f}s")
        print(f"  R²={metrics['r2']:.4f} | R²_log={metrics['r2_log']:.4f} | MAE={metrics['mae']:.4f} | RMSE={metrics['rmse']:.4f} | Macro Rel-MAE={metrics['macro_crop_rel_mae_pct']:.2f}%")

        results.append({
            "variant": v_id,
            "name": v_name,
            "description": spec["description"],
            "features": spec["features"],
            "time_based_r2": metrics["r2"],
            "time_based_r2_log": metrics["r2_log"],
            "time_based_mae": metrics["mae"],
            "time_based_rmse": metrics["rmse"],
            "macro_crop_mae_over_mean_yield_pct": metrics["macro_crop_rel_mae_pct"],
            "pipeline_artifact": f"models/variants/{variant_filename}",
            "per_crop_breakdown": metrics["per_crop_breakdown"],
        })

    # Save ablation.json
    output_payload = {
        "study": "AgriPredict AI Feature Ablation Study",
        "split": {
            "strategy": "Time-based temporal holdout",
            "train_years": "1997-2012",
            "test_years": "2013-2015",
            "train_samples": len(train_df),
            "test_samples": len(test_df),
        },
        "hist_gradient_boosting_hyperparameters": TUNED_HGB_PARAMS,
        "variants": results,
    }

    with open(ABLATION_JSON_PATH, "w") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\nAblation metrics saved to: {ABLATION_JSON_PATH.relative_to(PROJECT_ROOT)}")

    # Print clean comparison table
    print("\n" + "=" * 98)
    print(f"{'FEATURE ABLATION STUDY RESULTS (TIME-BASED SPLIT: TRAIN 1997-2012, TEST 2013-2015)':^98}")
    print("=" * 98)
    header = (
        f"{'Variant':<8} | {'Description':<38} | "
        f"{'R²':<7} | {'R²_log':<7} | {'MAE':<7} | {'RMSE':<7} | {'Macro Rel-MAE (%)':<17}"
    )
    print(header)
    print("-" * 98)
    for r in results:
        rel_mae_str = f"{r['macro_crop_mae_over_mean_yield_pct']:.2f}%"
        row = (
            f"{r['variant']:<8} | {r['name']:<38} | "
            f"{r['time_based_r2']:<7.4f} | {r['time_based_r2_log']:<7.4f} | "
            f"{r['time_based_mae']:<7.4f} | {r['time_based_rmse']:<7.4f} | "
            f"{rel_mae_str:<17}"
        )
        print(row)
    print("=" * 98)


if __name__ == "__main__":
    run_ablation()
