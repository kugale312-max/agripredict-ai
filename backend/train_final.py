"""
backend/train_final.py
======================
Final Production Model Training for AgriPredict AI.

Trains Variant C architecture across all historical years (1997-2015):
  - Features: Crop, Season, Rainfall_mm, Temperature_C, State (no Soil_Type, no Area_ha)
  - Preprocessing: StandardScaler for numericals, OneHotEncoder for categoricals
  - Architecture: HistGradientBoostingRegressor (tuned hyperparameters)
  - Target Transformation: TransformedTargetRegressor (log1p / expm1)
  - Target: Yield_t_per_ha

Saves:
  - models/best_model.joblib (full production pipeline)
  - models/metadata.json (rich metadata with validation metrics from robustness.json,
    valid_combos (>= 20 samples), typical_weather, and historical avg_yield)
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

RANDOM_STATE = 42
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

np.random.seed(RANDOM_STATE)

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "crop_yield_clean.csv"
MODELS_DIR = PROJECT_ROOT / "models"
ROBUSTNESS_PATH = MODELS_DIR / "robustness.json"
METADATA_PATH = MODELS_DIR / "metadata.json"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"

# Tuned HistGradientBoosting hyperparameters
TUNED_HGB_PARAMS = {
    "max_iter": 150,
    "max_depth": None,
    "learning_rate": 0.08,
    "min_samples_leaf": 40,
    "random_state": RANDOM_STATE,
}

FEATURE_COLS = ["Crop", "Season", "Rainfall_mm", "Temperature_C", "State"]
NUM_COLS = ["Rainfall_mm", "Temperature_C"]
CAT_COLS = ["Crop", "Season", "State"]


def load_and_filter_data(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at '{data_path}'.")

    df = pd.read_csv(data_path)
    initial_rows = len(df)

    # 1. Drop generic crops with 'other' in the name
    other_mask = df["Crop"].str.contains("other", case=False, na=False)
    df = df[~other_mask].copy()

    # 2. Drop minor crops with < 300 rows
    crop_counts = df["Crop"].value_counts()
    low_crops = crop_counts[crop_counts < 300]
    df = df[~df["Crop"].isin(low_crops.index)].copy()

    # 3. Drop rows with Yield_t_per_ha < 0.01 as anomalous errors
    low_yield_mask = df["Yield_t_per_ha"] < 0.01
    df = df[~low_yield_mask].copy()

    print(f"Data Filtered: {initial_rows:,} -> {len(df):,} rows retained across {df['Crop'].nunique()} crops and {df['State'].nunique()} states.")
    return df


def build_pipeline() -> TransformedTargetRegressor:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_COLS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CAT_COLS,
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


def extract_validation_metrics() -> Dict[str, Any]:
    """Reads rolling-window validation metrics from models/robustness.json."""
    if not ROBUSTNESS_PATH.exists():
        print(f"Warning: {ROBUSTNESS_PATH} not found. Validation metrics will be empty.")
        return {}

    with open(ROBUSTNESS_PATH, "r") as f:
        rob_data = json.load(f)

    # Extract 3-window means
    c_mean = rob_data.get("mean_over_windows", {}).get("C", {})
    d_mean = rob_data.get("mean_over_windows", {}).get("D", {})

    # Extract 2013-2015 window
    w_2013 = next((w for w in rob_data.get("windows", []) if w.get("window") == "2013-2015"), {})
    c_2013 = w_2013.get("variants", {}).get("C", {})
    d_2013 = w_2013.get("variants", {}).get("D", {})

    return {
        "evaluation_strategy": "Temporal Rolling Window Holdout (Strict Future Holdout, NOT in-sample)",
        "mean_over_three_rolling_windows_2007_2015": {
            "model_variant_C": {
                "description": "HistGradientBoosting (State-aware, no Soil Type, no Area)",
                "mae": c_mean.get("mean_mae"),
                "r2_log": c_mean.get("mean_r2_log"),
                "macro_crop_rel_mae_pct": c_mean.get("mean_macro_crop_rel_mae_pct"),
            },
            "baseline_variant_D": {
                "description": "Historical Crop x Season Mean Baseline",
                "mae": d_mean.get("mean_mae"),
                "r2_log": d_mean.get("mean_r2_log"),
                "macro_crop_rel_mae_pct": d_mean.get("mean_macro_crop_rel_mae_pct"),
            },
        },
        "latest_window_2013_2015": {
            "model_variant_C": {
                "description": "HistGradientBoosting (State-aware, no Soil Type, no Area)",
                "mae": c_2013.get("mae"),
                "r2_log": c_2013.get("r2_log"),
                "macro_crop_rel_mae_pct": c_2013.get("macro_crop_rel_mae_pct"),
            },
            "baseline_variant_D": {
                "description": "Historical Crop x Season Mean Baseline",
                "mae": d_2013.get("mae"),
                "r2_log": d_2013.get("r2_log"),
                "macro_crop_rel_mae_pct": d_2013.get("macro_crop_rel_mae_pct"),
            },
        },
    }


def compute_metadata(df: pd.DataFrame) -> Dict[str, Any]:
    print("Computing domain constraints, typical weather, and yield averages...")

    # 1. Valid Combos: (State, Crop, Season) with at least 20 rows
    combo_counts = df.groupby(["State", "Crop", "Season"]).size()
    valid_combos: Dict[str, Dict[str, List[str]]] = {}
    valid_combo_count = 0

    for (state, crop, season), count in combo_counts.items():
        if count >= 20:
            valid_combos.setdefault(state, {}).setdefault(crop, []).append(season)
            valid_combo_count += 1

    # Sort crops and seasons for clean alphabetical presentation
    for st in valid_combos:
        for cr in valid_combos[st]:
            valid_combos[st][cr] = sorted(valid_combos[st][cr])

    states_list = sorted(valid_combos.keys())

    # 2. Typical Weather: computed over unique (State, Crop_Year, Season) rows
    weather_unique = df[["State", "Crop_Year", "Season", "Rainfall_mm", "Temperature_C"]].drop_duplicates()
    typical_weather: Dict[str, Dict[str, Dict[str, float]]] = {}

    for (st, se), grp in weather_unique.groupby(["State", "Season"]):
        typical_weather.setdefault(st, {})[se] = {
            "rain_mean": round(float(grp["Rainfall_mm"].mean()), 1),
            "rain_min": round(float(grp["Rainfall_mm"].min()), 1),
            "rain_max": round(float(grp["Rainfall_mm"].max()), 1),
            "temp_mean": round(float(grp["Temperature_C"].mean()), 1),
            "temp_min": round(float(grp["Temperature_C"].min()), 1),
            "temp_max": round(float(grp["Temperature_C"].max()), 1),
        }

    # 3. Average Yield: State-Crop-Season for valid combos, plus National average per crop per season
    avg_yield_state: Dict[str, Dict[str, Dict[str, float]]] = {}
    for st, crops_dict in valid_combos.items():
        for cr, seasons in crops_dict.items():
            for se in seasons:
                sub = df[(df["State"] == st) & (df["Crop"] == cr) & (df["Season"] == se)]
                if len(sub) > 0:
                    y_mean = round(float(sub["Yield_t_per_ha"].mean()), 4)
                    avg_yield_state.setdefault(st, {}).setdefault(cr, {})[se] = y_mean

    national_avg_yield: Dict[str, Dict[str, float]] = {}
    for (cr, se), grp in df.groupby(["Crop", "Season"]):
        if len(grp) >= 20:
            national_avg_yield.setdefault(cr, {})[se] = round(float(grp["Yield_t_per_ha"].mean()), 4)

    val_metrics = extract_validation_metrics()

    metadata = {
        "model_name": "HistGradientBoostingRegressor (State-Aware)",
        "features": FEATURE_COLS,
        "trained_on": "1997-2015",
        "total_training_samples": len(df),
        "hard_limits": {
            "rainfall_min": 0.0,
            "rainfall_max": 5000.0,
            "temperature_min": -5.0,
            "temperature_max": 45.0,
            "area_min": 0.1,
            "area_max": 10000.0,
        },
        "validation_metrics": val_metrics,
        "states": states_list,
        "valid_combos": valid_combos,
        "total_valid_combos": valid_combo_count,
        "typical_weather": typical_weather,
        "avg_yield": avg_yield_state,
        "national_avg_yield": national_avg_yield,
    }

    return metadata


def main():
    start_time = time.time()
    print("=" * 80)
    print("AgriPredict AI: Final Production Model Training (Variant C: State-Aware)")
    print("=" * 80)

    # 1. Load and filter full dataset (1997-2015)
    df = load_and_filter_data(DATA_PATH)

    # 2. Build and fit pipeline on all rows
    print("\nFitting final HistGradientBoosting pipeline on full dataset (1997-2015)...")
    pipeline = build_pipeline()
    X = df[FEATURE_COLS]
    y = df["Yield_t_per_ha"]

    fit_start = time.time()
    pipeline.fit(X, y)
    print(f"Model fitting completed in {time.time() - fit_start:.1f}s.")

    # 3. Save production model
    joblib.dump(pipeline, BEST_MODEL_PATH)
    print(f"Saved production model to: {BEST_MODEL_PATH.name}")

    # 4. Generate and save metadata.json
    metadata = compute_metadata(df)
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved rich metadata to: {METADATA_PATH.name}")

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY & ARTIFACT METRICS")
    print("=" * 80)
    print(f"Trained On                  : 1997-2015 ({len(df):,} rows)")
    print(f"Number of States            : {len(metadata['states'])}")
    print(f"Number of Valid Combos (>=20): {metadata['total_valid_combos']:,}")
    print(f"Model File Timestamp        : {time.ctime(BEST_MODEL_PATH.stat().st_mtime)}")
    print(f"Metadata File Timestamp     : {time.ctime(METADATA_PATH.stat().st_mtime)}")
    print(f"Total Execution Time        : {total_time:.1f}s")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
