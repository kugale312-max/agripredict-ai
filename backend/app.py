"""
backend/app.py
==============
Flask REST API for AgriPredict AI Model Serving (Variant C: State-Aware).

Endpoints:
  - GET  /health   -> Health check status
  - GET  /options  -> States, valid_combos (State -> Crop -> Seasons), typical_weather, hard_limits
  - GET  /trends   -> Historical average yield per valid season for (Crop, State) or national average
  - GET  /metrics  -> Rolling-window temporal validation metrics from metadata.json
  - POST /predict  -> Predicts crop yield (t/ha) and total production (tonnes) with validation & warnings
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)

import joblib
import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, request, send_from_directory
from flask_cors import CORS

# File paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "best_model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"
PUBLIC_DIR = BASE_DIR / "public"

app = Flask(__name__)
CORS(app)

# Global containers
MODEL = None
METADATA: Dict[str, Any] = {}
EXPECTED_FEATURES: List[str] = []
MODEL_NAME = "HistGradientBoostingRegressor (State-Aware)"


def load_artifacts():
    global MODEL, METADATA, EXPECTED_FEATURES, MODEL_NAME

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file missing: '{MODEL_PATH}'. Run backend/train_final.py first.")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file missing: '{METADATA_PATH}'. Run backend/train_final.py first.")

    try:
        MODEL = joblib.load(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from '{MODEL_PATH}': {e}") from e

    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            METADATA = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load metadata from '{METADATA_PATH}': {e}") from e

    if hasattr(MODEL, "feature_names_in_"):
        EXPECTED_FEATURES = list(MODEL.feature_names_in_)
    elif hasattr(MODEL, "regressor_") and hasattr(MODEL.regressor_, "feature_names_in_"):
        EXPECTED_FEATURES = list(MODEL.regressor_.feature_names_in_)
    else:
        EXPECTED_FEATURES = ["Crop", "Season", "Rainfall_mm", "Temperature_C", "State"]

    MODEL_NAME = METADATA.get("model_name", "HistGradientBoostingRegressor (State-Aware)")

    print("=" * 70)
    print("AgriPredict AI Model Server Initialized")
    print(f"Model File       : {MODEL_PATH.name}")
    print(f"Model Name       : {MODEL_NAME}")
    print(f"Expected Features: {EXPECTED_FEATURES}")
    print(f"States Available : {len(METADATA.get('states', []))}")
    print("=" * 70)


load_artifacts()


# =============================================================================
# 1. Health Check
# =============================================================================
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "model": MODEL_NAME}), 200


# =============================================================================
# 2. Options Endpoint
# =============================================================================
@app.route("/options", methods=["GET"])
def get_options():
    """
    Returns available states, valid State-Crop-Season combinations,
    typical seasonal weather statistics, and input hard limits.
    """
    payload = {
        "states": METADATA.get("states", []),
        "valid_combos": METADATA.get("valid_combos", {}),
        "typical_weather": METADATA.get("typical_weather", {}),
        "hard_limits": METADATA.get("hard_limits", {
            "rainfall_min": 0.0,
            "rainfall_max": 5000.0,
            "temperature_min": -5.0,
            "temperature_max": 45.0,
            "area_min": 0.1,
            "area_max": 10000.0,
        }),
    }
    return jsonify(payload), 200


# =============================================================================
# 3. Trends Endpoint
# =============================================================================
@app.route("/trends", methods=["GET"])
def get_trends():
    """
    Returns historical average yield per valid season for a crop in a state.
    If state is omitted, falls back to the national average per crop per season.
    Returns 400 for unknown/invalid values.
    """
    crop = request.args.get("crop")
    state = request.args.get("state")

    if not crop:
        return jsonify({"error": "Missing required query parameter: 'crop'"}), 400

    valid_combos = METADATA.get("valid_combos", {})
    national_avg = METADATA.get("national_avg_yield", {})
    state_avg = METADATA.get("avg_yield", {})

    all_crops = sorted({c for crops in valid_combos.values() for c in crops})

    if crop not in all_crops and crop not in national_avg:
        return jsonify({
            "error": f"Unknown crop '{crop}'. Valid crops are: {', '.join(all_crops[:15])}..."
        }), 400

    if state:
        if state not in valid_combos:
            valid_states_str = ", ".join(sorted(valid_combos.keys()))
            return jsonify({
                "error": f"Invalid state '{state}'. Valid states are: {valid_states_str}"
            }), 400

        crops_in_state = valid_combos.get(state, {})
        if crop not in crops_in_state:
            valid_crops_str = ", ".join(sorted(crops_in_state.keys()))
            return jsonify({
                "error": f"Crop '{crop}' is not valid for state '{state}'. Valid crops in {state} are: {valid_crops_str}"
            }), 400

        valid_seasons = crops_in_state[crop]
        seasonal_averages = state_avg.get(state, {}).get(crop, {})
        trends_list = [
            {"season": s, "average_yield": round(float(seasonal_averages.get(s, 0.0)), 2)}
            for s in valid_seasons
            if s in seasonal_averages
        ]

        return jsonify({
            "crop": crop,
            "state": state,
            "level": "state",
            "valid_seasons": valid_seasons,
            "seasonal_averages": {s: round(float(seasonal_averages[s]), 2) for s in seasonal_averages},
            "trends": trends_list,
        }), 200

    # National average fallback
    if crop not in national_avg:
        return jsonify({
            "error": f"No national average yield data available for crop '{crop}'."
        }), 400

    crop_nat = national_avg[crop]
    valid_seasons = sorted(crop_nat.keys())
    trends_list = [
        {"season": s, "average_yield": round(float(crop_nat[s]), 2)}
        for s in valid_seasons
    ]

    return jsonify({
        "crop": crop,
        "state": None,
        "level": "national",
        "valid_seasons": valid_seasons,
        "seasonal_averages": {s: round(float(crop_nat[s]), 2) for s in crop_nat},
        "trends": trends_list,
    }), 200


# =============================================================================
# 4. Metrics Endpoint
# =============================================================================
@app.route("/metrics", methods=["GET"])
def get_metrics():
    """Returns the rolling-window validation metrics from metadata.json."""
    return jsonify({
        "model_name": METADATA.get("model_name", MODEL_NAME),
        "features": METADATA.get("features", EXPECTED_FEATURES),
        "trained_on": METADATA.get("trained_on", "1997-2015"),
        "validation_metrics": METADATA.get("validation_metrics", {}),
    }), 200


# =============================================================================
# 5. Predict Endpoint
# =============================================================================
@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON: {state, crop, season, rainfall, temperature, area}.
    Validates:
      - Field presence and types
      - (State, Crop, Season) combination validity against valid_combos
      - Hard limits (400 if exceeded):
          rainfall: 0 - 5000 mm
          temperature: -5 - 45 °C
          area: 0.1 - 10000 ha
      - Seen min-max range check (adds warning message if outside state's historical seen range)
    Builds DataFrame matching MODEL feature_names_in_.
    Returns:
      {
        "predicted_yield": float,
        "total_production": float,
        "model_name": str,
        "warnings": List[str],
        "note": str
      }
    """
    if not request.is_json:
        return jsonify({"error": "Request body must be valid JSON with 'Content-Type: application/json'"}), 400

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request payload must be a JSON object"}), 400

    # 1. Required fields
    required_fields = ["state", "crop", "season", "rainfall", "temperature", "area"]
    for field in required_fields:
        if field not in data or data[field] is None or data[field] == "":
            return jsonify({"error": f"Missing required field: '{field}'"}), 400

    state = str(data["state"]).strip()
    crop = str(data["crop"]).strip()
    season = str(data["season"]).strip()

    # 2. Parse numerical types
    try:
        rainfall = float(data["rainfall"])
    except (ValueError, TypeError):
        return jsonify({"error": "Field 'rainfall' must be a valid number (mm)."}), 400

    try:
        temperature = float(data["temperature"])
    except (ValueError, TypeError):
        return jsonify({"error": "Field 'temperature' must be a valid number (°C)."}), 400

    try:
        area = float(data["area"])
    except (ValueError, TypeError):
        return jsonify({"error": "Field 'area' must be a valid number (hectares)."}), 400

    # 3. Validate combination against valid_combos
    valid_combos = METADATA.get("valid_combos", {})
    if state not in valid_combos:
        valid_states_str = ", ".join(sorted(valid_combos.keys()))
        return jsonify({
            "error": f"Invalid state '{state}'. Valid states with sufficient historical data are: {valid_states_str}"
        }), 400

    crops_in_state = valid_combos[state]
    if crop not in crops_in_state:
        valid_crops_str = ", ".join(sorted(crops_in_state.keys()))
        return jsonify({
            "error": f"Invalid crop '{crop}' for state '{state}'. Valid crops in {state} are: {valid_crops_str}"
        }), 400

    seasons_for_crop = crops_in_state[crop]
    if season not in seasons_for_crop:
        valid_seasons_str = ", ".join(seasons_for_crop)
        return jsonify({
            "error": f"Invalid season '{season}' for {crop} in {state}. Valid seasons are: {valid_seasons_str}"
        }), 400

    # 4. Hard limits validation (400 if exceeded)
    hard_limits = METADATA.get("hard_limits", {})
    r_min = hard_limits.get("rainfall_min", 0.0)
    r_max = hard_limits.get("rainfall_max", 5000.0)
    t_min = hard_limits.get("temperature_min", -5.0)
    t_max = hard_limits.get("temperature_max", 45.0)
    a_min = hard_limits.get("area_min", 0.1)
    a_max = hard_limits.get("area_max", 10000.0)

    if rainfall < r_min or rainfall > r_max:
        return jsonify({
            "error": f"Rainfall {rainfall} mm is outside the hard limit [{r_min}, {r_max}] mm."
        }), 400

    if temperature < t_min or temperature > t_max:
        return jsonify({
            "error": f"Temperature {temperature} °C is outside the hard limit [{t_min}, {t_max}] °C."
        }), 400

    if area < a_min or area > a_max:
        return jsonify({
            "error": f"Area {area} ha is outside the hard limit [{a_min}, {a_max}] ha."
        }), 400

    # 5. Seen min-max range check (adds warnings, still predicts)
    warnings: List[str] = []
    typical_weather = METADATA.get("typical_weather", {}).get(state, {}).get(season, {})
    if typical_weather:
        seen_rain_min = typical_weather.get("rain_min")
        seen_rain_max = typical_weather.get("rain_max")
        seen_temp_min = typical_weather.get("temp_min")
        seen_temp_max = typical_weather.get("temp_max")

        if seen_rain_min is not None and seen_rain_max is not None:
            if rainfall < seen_rain_min or rainfall > seen_rain_max:
                warnings.append(
                    f"Rainfall ({rainfall:.1f} mm) is outside the historical observed range for {state} in {season} "
                    f"({seen_rain_min:.1f} to {seen_rain_max:.1f} mm). The yield estimate may be less reliable."
                )

        if seen_temp_min is not None and seen_temp_max is not None:
            if temperature < seen_temp_min or temperature > seen_temp_max:
                warnings.append(
                    f"Temperature ({temperature:.1f} °C) is outside the historical observed range for {state} in {season} "
                    f"({seen_temp_min:.1f} to {seen_temp_max:.1f} °C). The yield estimate may be less reliable."
                )

    # 6. Build DataFrame matching the pipeline's feature names exactly
    feature_mapping = {
        "Crop": crop,
        "Season": season,
        "Rainfall_mm": rainfall,
        "Temperature_C": temperature,
        "State": state,
    }

    input_row = {col: feature_mapping.get(col, None) for col in EXPECTED_FEATURES}
    input_df = pd.DataFrame([input_row], columns=EXPECTED_FEATURES)

    # 7. Model inference
    try:
        raw_pred = float(MODEL.predict(input_df)[0])
    except Exception as e:
        return jsonify({"error": f"Model inference error: {str(e)}"}), 500

    # Safeguard: Never return negative yield
    predicted_yield = round(max(0.0, raw_pred), 2)
    total_production = round(predicted_yield * area, 2)

    return jsonify({
        "predicted_yield": predicted_yield,
        "total_production": total_production,
        "model_name": MODEL_NAME,
        "warnings": warnings,
        "note": "Estimate based on historical district-level data; typical error is about 30% of a crop's average yield.",
    }), 200


# =============================================================================
# 6. Static Serving
# =============================================================================
@app.route("/", methods=["GET"])
def serve_index():
    if not (PUBLIC_DIR / "index.html").is_file():
        abort(404)
    return send_from_directory(str(PUBLIC_DIR), "index.html")


@app.route("/<path:filename>", methods=["GET"])
def serve_static(filename: str):
    file_path = PUBLIC_DIR / filename
    if not file_path.is_file():
        abort(404)
    return send_from_directory(str(PUBLIC_DIR), filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
