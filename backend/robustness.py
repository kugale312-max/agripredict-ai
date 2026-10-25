"""
backend/robustness.py
=====================
Temporal Robustness and Rolling-Window Evaluation for AgriPredict AI.

Evaluates variants across three expanding historical training / rolling test windows:
  - Window 1: Train 1997-2006 | Test 2007-2009
  - Window 2: Train 1997-2009 | Test 2010-2012
  - Window 3: Train 1997-2012 | Test 2013-2015

Variants:
  - Variant B: Crop, Season, Rainfall_mm, Temperature_C, Soil_Type
  - Variant C: Crop, Season, Rainfall_mm, Temperature_C, State
  - Variant D: Crop x Season mean baseline

For Variant C, computes the number of test rows with an unseen (State, Crop) pair.
Reports MAE, R²_log, and macro-average of per-crop relative MAE (%).
Saves results to models/robustness.json without touching models/best_model.joblib.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Environment configuration
RANDOM_STATE = 42
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if __name__ == "__main__":
    sys.modules["backend.robustness"] = sys.modules["__main__"]

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

np.random.seed(RANDOM_STATE)

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "crop_yield_clean.csv"
MODELS_DIR = PROJECT_ROOT / "models"
ROBUSTNESS_JSON_PATH = MODELS_DIR / "robustness.json"

# Fixed tuned HistGradientBoosting hyperparameters from Stage 2 (train_model.py)
TUNED_HGB_PARAMS = {
    "max_iter": 150,
    "max_depth": None,
    "learning_rate": 0.08,
    "min_samples_leaf": 40,
    "random_state": RANDOM_STATE,
}

# Windows specification
WINDOWS = [
    {
        "id": "W1",
        "name": "2007-2009",
        "train_years": (1997, 2006),
        "test_years": (2007, 2009),
    },
    {
        "id": "W2",
        "name": "2010-2012",
        "train_years": (1997, 2009),
        "test_years": (2010, 2012),
    },
    {
        "id": "W3",
        "name": "2013-2015",
        "train_years": (1997, 2012),
        "test_years": (2013, 2015),
    },
]


# =============================================================================
# Custom Estimator: Naive Crop x Season Mean Baseline (Variant D)
# =============================================================================
class NaiveCropSeasonMeanRegressor(BaseEstimator, RegressorMixin):
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
    if not data_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at '{data_path}'.")

    df = pd.read_csv(data_path)
    initial_rows = len(df)

    # 1. Drop generic crops with 'other'
    other_mask = df["Crop"].str.contains("other", case=False, na=False)
    df = df[~other_mask].copy()

    # 2. Drop minor crops with < 300 rows
    crop_counts = df["Crop"].value_counts()
    low_crops = crop_counts[crop_counts < 300]
    df = df[~df["Crop"].isin(low_crops.index)].copy()

    # 3. Drop rows with Yield_t_per_ha < 0.01
    low_yield_mask = df["Yield_t_per_ha"] < 0.01
    df = df[~low_yield_mask].copy()

    print(f"Data Filtered: {initial_rows:,} -> {len(df):,} rows retained across {df['Crop'].nunique()} crops.")
    return df


# =============================================================================
# Pipeline Builder
# =============================================================================
def build_model_pipeline(num_cols: List[str], cat_cols: List[str]) -> TransformedTargetRegressor:
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
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, crop_series: pd.Series) -> Dict[str, float]:
    y_pred_clipped = np.maximum(y_pred, 0)
    mae = float(mean_absolute_error(y_true, y_pred))
    r2_log = float(r2_score(np.log1p(y_true), np.log1p(y_pred_clipped)))

    eval_df = pd.DataFrame({
        "Crop": crop_series.values,
        "y_true": y_true,
        "y_pred": y_pred,
    })

    rel_maes = []
    for _, grp in eval_df.groupby("Crop"):
        c_true = grp["y_true"].values
        c_pred = grp["y_pred"].values
        c_mae = float(mean_absolute_error(c_true, c_pred))
        c_mean = float(np.mean(c_true))
        c_rel = (c_mae / c_mean) if c_mean > 0 else 0.0
        rel_maes.append(c_rel)

    macro_rel_mae_pct = float(np.mean(rel_maes) * 100)

    return {
        "mae": round(mae, 4),
        "r2_log": round(r2_log, 4),
        "macro_crop_rel_mae_pct": round(macro_rel_mae_pct, 2),
    }


# =============================================================================
# Main Robustness Runner
# =============================================================================
def run_robustness():
    print("=" * 85)
    print("AgriPredict AI: Rolling-Window Temporal Robustness Evaluation")
    print(f"HistGradientBoosting Parameters: {TUNED_HGB_PARAMS}")
    print("=" * 85)

    df = load_and_filter_data(DATA_PATH)

    variants_config = {
        "B": {
            "name": "Variant B (Soil_Type, No Area)",
            "features": ["Crop", "Season", "Rainfall_mm", "Temperature_C", "Soil_Type"],
            "num_cols": ["Rainfall_mm", "Temperature_C"],
            "cat_cols": ["Crop", "Season", "Soil_Type"],
            "is_baseline": False,
        },
        "C": {
            "name": "Variant C (State, No Soil_Type)",
            "features": ["Crop", "Season", "Rainfall_mm", "Temperature_C", "State"],
            "num_cols": ["Rainfall_mm", "Temperature_C"],
            "cat_cols": ["Crop", "Season", "State"],
            "is_baseline": False,
        },
        "D": {
            "name": "Variant D (Crop x Season Baseline)",
            "features": ["Crop", "Season"],
            "num_cols": [],
            "cat_cols": [],
            "is_baseline": True,
        },
    }

    window_results = []
    unseen_state_crop_stats = []

    for w in WINDOWS:
        w_name = w["name"]
        tr_min, tr_max = w["train_years"]
        te_min, te_max = w["test_years"]

        train_mask = (df["Crop_Year"] >= tr_min) & (df["Crop_Year"] <= tr_max)
        test_mask = (df["Crop_Year"] >= te_min) & (df["Crop_Year"] <= te_max)

        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()

        y_train = train_df["Yield_t_per_ha"]
        y_test = test_df["Yield_t_per_ha"].values
        crop_test = test_df["Crop"]

        print(f"\n" + "-" * 75)
        print(f"Window: Test {w_name} | Train {tr_min}-{tr_max}")
        print(f"  Train samples: {len(train_df):,} | Test samples: {len(test_df):,}")
        print("-" * 75)

        # Unseen (State, Crop) calculation for Variant C
        train_pairs = set(zip(train_df["State"], train_df["Crop"]))
        test_pairs = list(zip(test_df["State"], test_df["Crop"]))
        unseen_count = sum(1 for p in test_pairs if p not in train_pairs)
        unseen_pct = (unseen_count / len(test_df)) * 100 if len(test_df) > 0 else 0.0

        unseen_state_crop_stats.append({
            "window": w_name,
            "train_years": f"{tr_min}-{tr_max}",
            "test_years": f"{te_min}-{te_max}",
            "total_test_rows": len(test_df),
            "unseen_state_crop_rows": unseen_count,
            "unseen_state_crop_pct": round(unseen_pct, 2),
        })

        print(f"  [Variant C Diagnostics] Unseen (State, Crop) test rows: {unseen_count:,} / {len(test_df):,} ({unseen_pct:.2f}%)")

        window_record = {
            "window": w_name,
            "train_years": f"{tr_min}-{tr_max}",
            "test_years": f"{te_min}-{te_max}",
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "variants": {},
        }

        for v_key in ["B", "C", "D"]:
            v_conf = variants_config[v_key]
            t0 = time.time()

            if v_conf["is_baseline"]:
                model = Pipeline([
                    ("naive_baseline", NaiveCropSeasonMeanRegressor())
                ])
                model.fit(train_df[v_conf["features"]], y_train)
                preds = model.predict(test_df[v_conf["features"]])
            else:
                model = build_model_pipeline(v_conf["num_cols"], v_conf["cat_cols"])
                model.fit(train_df[v_conf["features"]], y_train)
                preds = model.predict(test_df[v_conf["features"]])

            elapsed = time.time() - t0
            metrics = compute_metrics(y_test, preds, crop_test)
            window_record["variants"][v_key] = metrics

            print(f"  Variant {v_key:<1} ({v_conf['name']:<31}): MAE={metrics['mae']:.4f} | R²_log={metrics['r2_log']:.4f} | Rel-MAE={metrics['macro_crop_rel_mae_pct']:.2f}% ({elapsed:.1f}s)")

        window_results.append(window_record)

    # Compute means over the 3 windows for each variant
    means_summary = {}
    for v_key in ["B", "C", "D"]:
        maes = [w["variants"][v_key]["mae"] for w in window_results]
        r2_logs = [w["variants"][v_key]["r2_log"] for w in window_results]
        rel_maes = [w["variants"][v_key]["macro_crop_rel_mae_pct"] for w in window_results]

        means_summary[v_key] = {
            "mean_mae": round(float(np.mean(maes)), 4),
            "mean_r2_log": round(float(np.mean(r2_logs)), 4),
            "mean_macro_crop_rel_mae_pct": round(float(np.mean(rel_maes)), 2),
        }

    # Print comprehensive comparison table
    print("\n" + "=" * 105)
    print(f"{'ROLLING-WINDOW ROBUSTNESS COMPARISON TABLE':^105}")
    print("=" * 105)
    header = (
        f"{'Window (Test)':<15} | "
        f"{'B MAE':<7} {'B R²_log':<8} {'B Rel%':<7} | "
        f"{'C MAE':<7} {'C R²_log':<8} {'C Rel%':<7} | "
        f"{'D MAE':<7} {'D R²_log':<8} {'D Rel%':<7}"
    )
    print(header)
    print("-" * 105)

    for w in window_results:
        vb = w["variants"]["B"]
        vc = w["variants"]["C"]
        vd = w["variants"]["D"]
        row_str = (
            f"{w['window']:<15} | "
            f"{vb['mae']:<7.4f} {vb['r2_log']:<8.4f} {vb['macro_crop_rel_mae_pct']:<7.2f} | "
            f"{vc['mae']:<7.4f} {vc['r2_log']:<8.4f} {vc['macro_crop_rel_mae_pct']:<7.2f} | "
            f"{vd['mae']:<7.4f} {vd['r2_log']:<8.4f} {vd['macro_crop_rel_mae_pct']:<7.2f}"
        )
        print(row_str)

    print("-" * 105)
    mb = means_summary["B"]
    mc = means_summary["C"]
    md = means_summary["D"]
    mean_row_str = (
        f"{'Mean (3 Windows)':<15} | "
        f"{mb['mean_mae']:<7.4f} {mb['mean_r2_log']:<8.4f} {mb['mean_macro_crop_rel_mae_pct']:<7.2f} | "
        f"{mc['mean_mae']:<7.4f} {mc['mean_r2_log']:<8.4f} {mc['mean_macro_crop_rel_mae_pct']:<7.2f} | "
        f"{md['mean_mae']:<7.4f} {md['mean_r2_log']:<8.4f} {md['mean_macro_crop_rel_mae_pct']:<7.2f}"
    )
    print(mean_row_str)
    print("=" * 105)

    # Print Unseen (State, Crop) summary for Variant C
    print("\n" + "=" * 80)
    print(f"{'VARIANT C: UNSEEN (STATE, CROP) COMBINATIONS IN TEST WINDOWS':^80}")
    print("=" * 80)
    print(f"{'Test Window':<15} | {'Training Window':<17} | {'Total Test Rows':<16} | {'Unseen Rows':<12} | {'Unseen %':<9}")
    print("-" * 80)
    for stat in unseen_state_crop_stats:
        print(
            f"{stat['window']:<15} | "
            f"{stat['train_years']:<17} | "
            f"{stat['total_test_rows']:<16,} | "
            f"{stat['unseen_state_crop_rows']:<12,} | "
            f"{stat['unseen_state_crop_pct']:<8.2f}%"
        )
    print("=" * 80)

    # Output JSON payload
    payload = {
        "study": "AgriPredict AI Temporal Robustness and Rolling-Window Study",
        "description": "Evaluation of variants B, C, and D across three rolling time windows.",
        "hist_gradient_boosting_hyperparameters": TUNED_HGB_PARAMS,
        "variants_spec": {
            "B": variants_config["B"]["name"],
            "C": variants_config["C"]["name"],
            "D": variants_config["D"]["name"],
        },
        "windows": window_results,
        "mean_over_windows": means_summary,
        "variant_c_unseen_state_crop_statistics": unseen_state_crop_stats,
    }

    with open(ROBUSTNESS_JSON_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nRobustness evaluation saved to: {ROBUSTNESS_JSON_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    run_robustness()
