"""
backend/app.py
==============
Flask REST API for AgriPredict AI Model Serving.

Provides RESTful endpoints for:
- Health check (/health)
- Dynamic options & input ranges (/options)
- Crop seasonal yield trends (/trends)
- Model performance metrics (/metrics)
- Yield prediction with strict validation (/predict)

Integrates:
- Flask and Flask-CORS for cross-origin access
- Pre-trained best_model.joblib and metadata.json loaded once at startup
- TransformedTargetRegressor inference with negative-yield safeguards
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Suppress loky CPU core warnings on Windows
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)

import joblib
import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, request, send_from_directory
from flask_cors import CORS

# File paths built strictly from file location
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "best_model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"
PUBLIC_DIR = BASE_DIR / "public"

# Initialize Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global containers for artifacts loaded once at startup
MODEL = None
METADATA = None
EXPECTED_FEATURES = []
MODEL_NAME = "HistGradientBoosting"


def load_artifacts():
    """
    Loads best_model.joblib and metadata.json once at application startup.
    Raises clear, descriptive FileNotFoundError if any artifact is missing.
    """
    global MODEL, METADATA, EXPECTED_FEATURES, MODEL_NAME

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Production model file missing: '{MODEL_PATH}'. "
            "Please run 'backend/train_model.py' to generate Stage 2 model artifacts before starting the API."
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata specification file missing: '{METADATA_PATH}'. "
            "Please ensure Stage 2 model training completed successfully."
        )

    try:
        MODEL = joblib.load(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to deserialize model artifact from '{MODEL_PATH}': {e}") from e

    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            METADATA = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to parse metadata JSON from '{METADATA_PATH}': {e}") from e

    # Determine input feature schema expected by the trained pipeline
    if hasattr(MODEL, "feature_names_in_"):
        EXPECTED_FEATURES = list(MODEL.feature_names_in_)
    elif hasattr(MODEL, "regressor_") and hasattr(MODEL.regressor_, "feature_names_in_"):
        EXPECTED_FEATURES = list(MODEL.regressor_.feature_names_in_)
    else:
        EXPECTED_FEATURES = ["Crop", "Season", "Rainfall_mm", "Temperature_C", "Soil_Type", "Area_ha"]

    # Extract model name
    best_model_meta = METADATA.get("best_model", {})
    MODEL_NAME = best_model_meta.get("architecture", "HistGradientBoosting")

    print("=" * 70)
    print("AgriPredict AI: Model Server Initialized Successfully")
    print(f"Model Artifact  : {MODEL_PATH.name}")
    print(f"Architecture    : {MODEL_NAME}")
    print(f"Model Features  : {EXPECTED_FEATURES}")
    print(f"Valid Crops     : {len(METADATA.get('valid_crops', []))} crops loaded")
    print("=" * 70)


# Trigger artifact loading on module import
load_artifacts()


# =============================================================================
# 1. Health Check Endpoint
# =============================================================================
@app.route("/health", methods=["GET"])
def health_check():
    """Returns service health status."""
    return jsonify({"status": "ok"}), 200


# =============================================================================
# 2. Options and Input Ranges Endpoint
# =============================================================================
@app.route("/options", methods=["GET"])
def get_options():
    """
    Returns valid crops, seasons, soil types, valid_crop_seasons mapping,
    and input ranges.
    - Rainfall & Temperature use 1st to 99th percentile ranges from metadata.json.
    - Area uses a fixed range of 0.1 to 10,000 hectares.
    """
    input_ranges_meta = METADATA.get("input_ranges", {})
    rain_meta = input_ranges_meta.get("Rainfall_mm", {})
    temp_meta = input_ranges_meta.get("Temperature_C", {})

    options_payload = {
        "crops": METADATA.get("valid_crops", []),
        "seasons": METADATA.get("valid_seasons", []),
        "soil_types": METADATA.get("valid_soil_types", []),
        "valid_crop_seasons": METADATA.get("valid_crop_seasons", {}),
        "input_ranges": {
            "rainfall": {
                "min": rain_meta.get("p01", 4.51),
                "max": rain_meta.get("p99", 1754.85),
                "unit": "mm",
            },
            "temperature": {
                "min": temp_meta.get("p01", 11.96),
                "max": temp_meta.get("p99", 32.39),
                "unit": "°C",
            },
            "area": {
                "min": 0.1,
                "max": 10000.0,
                "unit": "ha",
            },
        },
    }
    return jsonify(options_payload), 200


# =============================================================================
# 3. Trends Endpoint
# =============================================================================
@app.route("/trends", methods=["GET"])
def get_trends():
    """
    Returns average yield per season for a specified crop from metadata.json,
    strictly restricted to the valid seasons for that crop.
    Returns 400 for an unknown or missing crop.
    """
    crop = request.args.get("crop")
    if not crop:
        return jsonify({"error": "Missing required query parameter: 'crop'"}), 400

    valid_crops = METADATA.get("valid_crops", [])
    if crop not in valid_crops:
        return jsonify({
            "error": f"Unknown crop '{crop}'. Valid crops are: {', '.join(valid_crops)}"
        }), 400

    valid_crop_seasons = METADATA.get("valid_crop_seasons", {}).get(crop, [])
    avg_yield_map = METADATA.get("average_yield_per_crop_season", {})

    seasonal_trends = {}
    trends_list = []
    for season in valid_crop_seasons:
        key = f"{crop}__{season}"
        avg_val = avg_yield_map.get(key)
        if avg_val is not None:
            rounded_val = round(float(avg_val), 2)
            seasonal_trends[season] = rounded_val
            trends_list.append({"season": season, "average_yield": rounded_val})

    return jsonify({
        "crop": crop,
        "valid_seasons": valid_crop_seasons,
        "seasonal_averages": seasonal_trends,
        "trends": trends_list,
    }), 200


# =============================================================================
# 4. Metrics Endpoint
# =============================================================================
@app.route("/metrics", methods=["GET"])
def get_metrics():
    """Returns the best model name along with time-based R², MAE, and RMSE."""
    best_model_meta = METADATA.get("best_model", {})
    time_metrics = best_model_meta.get("time_based_metrics", {})

    metrics_payload = {
        "model_name": best_model_meta.get("architecture", MODEL_NAME),
        "time_based_r2": time_metrics.get("R2"),
        "time_based_r2_log": time_metrics.get("R2_log"),
        "time_based_mae": time_metrics.get("MAE"),
        "time_based_rmse": time_metrics.get("RMSE"),
    }
    return jsonify(metrics_payload), 200


# =============================================================================
# 5. Predict Endpoint
# =============================================================================
@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON: {crop, season, rainfall, temperature, soil, area}.
    Validates:
      - Field presence and JSON format
      - Valid categories against metadata
      - Valid (crop, season) pairing against valid_crop_seasons
      - Value ranges (percentiles for rainfall & temperature, 0.1-10000 ha for area)
    Builds DataFrame matching MODEL feature_names_in_.
    Returns:
      {
        "predicted_yield": float (t/ha, 2 decimals, non-negative),
        "total_production": float (tonnes, 2 decimals),
        "model_name": str,
        "note": str
      }
    """
    if not request.is_json:
        return jsonify({"error": "Request body must be valid JSON with 'Content-Type: application/json'"}), 400

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request payload must be a JSON object"}), 400

    # 1. Check required fields
    required_fields = ["crop", "season", "rainfall", "temperature", "soil", "area"]
    for field in required_fields:
        if field not in data or data[field] is None or data[field] == "":
            return jsonify({"error": f"Missing required field: '{field}'"}), 400

    raw_crop = data["crop"]
    raw_season = data["season"]
    raw_soil = data["soil"]
    raw_rainfall = data["rainfall"]
    raw_temp = data["temperature"]
    raw_area = data["area"]

    # 2. Validate categorical: Crop
    if not isinstance(raw_crop, str):
        return jsonify({"error": "Field 'crop' must be a string"}), 400
    crop = raw_crop.strip()
    valid_crops = METADATA.get("valid_crops", [])
    if crop not in valid_crops:
        return jsonify({
            "error": f"Invalid crop '{crop}'. Valid options are: {', '.join(valid_crops)}"
        }), 400

    # 3. Validate categorical: Season
    if not isinstance(raw_season, str):
        return jsonify({"error": "Field 'season' must be a string"}), 400
    season = raw_season.strip()
    valid_seasons = METADATA.get("valid_seasons", [])
    if season not in valid_seasons:
        return jsonify({
            "error": f"Invalid season '{season}'. Valid options are: {', '.join(valid_seasons)}"
        }), 400

    # 4. Validate Crop and Season combination
    valid_crop_seasons = METADATA.get("valid_crop_seasons", {})
    allowed_seasons_for_crop = valid_crop_seasons.get(crop, [])
    if season not in allowed_seasons_for_crop:
        return jsonify({
            "error": (
                f"Invalid crop and season combination: '{crop}' is not typically cultivated in '{season}' season. "
                f"Valid seasons for '{crop}' are: {', '.join(allowed_seasons_for_crop)}"
            )
        }), 400

    # 5. Validate categorical: Soil Type
    if not isinstance(raw_soil, str):
        return jsonify({"error": "Field 'soil' must be a string"}), 400
    soil = raw_soil.strip()
    valid_soil_types = METADATA.get("valid_soil_types", [])
    if soil not in valid_soil_types:
        return jsonify({
            "error": f"Invalid soil type '{soil}'. Valid options are: {', '.join(valid_soil_types)}"
        }), 400

    # 6. Validate numerical: Rainfall (percentile range from metadata)
    try:
        rainfall = float(raw_rainfall)
    except (ValueError, TypeError):
        return jsonify({"error": f"Field 'rainfall' must be a numeric value, got '{raw_rainfall}'"}), 400

    rain_meta = METADATA.get("input_ranges", {}).get("Rainfall_mm", {})
    rain_min = float(rain_meta.get("p01", 4.51))
    rain_max = float(rain_meta.get("p99", 1754.85))
    if rainfall < rain_min or rainfall > rain_max:
        return jsonify({
            "error": f"Rainfall value {rainfall} mm is out of range. Allowed range: {rain_min} to {rain_max} mm."
        }), 400

    # 7. Validate numerical: Temperature (percentile range from metadata)
    try:
        temperature = float(raw_temp)
    except (ValueError, TypeError):
        return jsonify({"error": f"Field 'temperature' must be a numeric value, got '{raw_temp}'"}), 400

    temp_meta = METADATA.get("input_ranges", {}).get("Temperature_C", {})
    temp_min = float(temp_meta.get("p01", 11.96))
    temp_max = float(temp_meta.get("p99", 32.39))
    if temperature < temp_min or temperature > temp_max:
        return jsonify({
            "error": f"Temperature value {temperature} °C is out of range. Allowed range: {temp_min} to {temp_max} °C."
        }), 400

    # 8. Validate numerical: Area (fixed range 0.1 to 10000 hectares)
    try:
        area = float(raw_area)
    except (ValueError, TypeError):
        return jsonify({"error": f"Field 'area' must be a numeric value, got '{raw_area}'"}), 400

    area_min = 0.1
    area_max = 10000.0
    if area < area_min or area > area_max:
        return jsonify({
            "error": f"Area value {area} ha is out of range. Allowed range: {area_min} to {area_max} hectares."
        }), 400

    # 9. Build one-row DataFrame using exactly the columns expected by the pipeline
    feature_mapping = {
        "Crop": crop,
        "Season": season,
        "Rainfall_mm": rainfall,
        "Temperature_C": temperature,
        "Soil_Type": soil,
        "Area_ha": area,
    }

    input_row = {}
    for col in EXPECTED_FEATURES:
        if col in feature_mapping:
            input_row[col] = feature_mapping[col]
        else:
            input_row[col] = None

    input_df = pd.DataFrame([input_row], columns=EXPECTED_FEATURES)

    # 10. Perform model inference
    try:
        raw_prediction = float(MODEL.predict(input_df)[0])
    except Exception as e:
        return jsonify({"error": f"Model inference error: {str(e)}"}), 500

    # Safeguard: Never return negative yield
    predicted_yield = round(max(0.0, raw_prediction), 2)
    total_production = round(predicted_yield * area, 2)

    return jsonify({
        "predicted_yield": predicted_yield,
        "total_production": total_production,
        "model_name": MODEL_NAME,
        "note": "This is an estimate based on historical district-level data.",
    }), 200



# =============================================================================
# 6. Static Frontend Routes
# =============================================================================
@app.route("/", methods=["GET"])
def serve_index():
    """Serves index.html from PUBLIC_DIR as fallback."""
    if not (PUBLIC_DIR / "index.html").is_file():
        abort(404)
    return send_from_directory(str(PUBLIC_DIR), "index.html")


@app.route("/<path:filename>", methods=["GET"])
def serve_static(filename: str):
    """
    Serves static assets (style.css, script.js, etc.) from PUBLIC_DIR as fallback.
    Returns 404 for missing files. API routes take priority over this route.
    """
    file_path = PUBLIC_DIR / filename
    if not file_path.is_file():
        abort(404)
    return send_from_directory(str(PUBLIC_DIR), filename)


# =============================================================================
# Server Entrypoint
# =============================================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
