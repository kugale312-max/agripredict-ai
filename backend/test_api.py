"""
backend/test_api.py
===================
Automated Test Suite for AgriPredict AI REST API (Variant C: State-Aware).

Tests:
  1. Valid prediction (Punjab, Wheat, Rabi, prefilled typical weather, Area=100)
  2. Invalid state-crop pair (returns 400)
  3. Invalid season for a valid state-crop pair (returns 400)
  4. Missing field (returns 400)
  5. Rainfall above hard limit (returns 400)
  6. Rainfall inside hard limits but outside state's seen range (returns 200 with warnings)
  7. Structure of /options endpoint
  8. Structure of /trends endpoint (state-level, national fallback, and error cases)
  9. Structure of /metrics endpoint
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app import app


def run_tests():
    client = app.test_client()
    passed = 0
    failed = 0

    def assert_test(condition, test_name, detail=""):
        nonlocal passed, failed
        if condition:
            print(f"  [PASS] {test_name}")
            passed += 1
        else:
            print(f"  [FAIL] {test_name} - {detail}")
            failed += 1

    print("=" * 80)
    print("AGRIPREDICT AI REST API AUTOMATED TEST SUITE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Test 1: Structure of /options
    # -------------------------------------------------------------------------
    print("\n--- Test Suite 1: /options Endpoint ---")
    res = client.get("/options")
    assert_test(res.status_code == 200, "GET /options returns HTTP 200")
    data = res.get_json()
    assert_test("states" in data and isinstance(data["states"], list), "Options contains 'states' list")
    assert_test("valid_combos" in data and isinstance(data["valid_combos"], dict), "Options contains 'valid_combos' dict")
    assert_test("typical_weather" in data and isinstance(data["typical_weather"], dict), "Options contains 'typical_weather' dict")
    assert_test("hard_limits" in data and isinstance(data["hard_limits"], dict), "Options contains 'hard_limits' dict")

    # -------------------------------------------------------------------------
    # Test 2: Structure of /trends
    # -------------------------------------------------------------------------
    print("\n--- Test Suite 2: /trends Endpoint ---")
    # State-level
    res = client.get("/trends?crop=Wheat&state=Punjab")
    assert_test(res.status_code == 200, "GET /trends?crop=Wheat&state=Punjab returns HTTP 200")
    data = res.get_json()
    assert_test(data.get("crop") == "Wheat" and data.get("state") == "Punjab", "Trends returns correct crop and state")
    assert_test("valid_seasons" in data and "Rabi" in data["valid_seasons"], "Wheat in Punjab includes Rabi season")
    assert_test("trends" in data and len(data["trends"]) > 0, "Trends contains seasonal trends array")

    # National-level fallback
    res_nat = client.get("/trends?crop=Wheat")
    assert_test(res_nat.status_code == 200, "GET /trends?crop=Wheat (national) returns HTTP 200")
    data_nat = res_nat.get_json()
    assert_test(data_nat.get("state") is None and data_nat.get("level") == "national", "National fallback level is set")

    # Missing crop
    res_err1 = client.get("/trends")
    assert_test(res_err1.status_code == 400, "GET /trends without crop returns HTTP 400")

    # Invalid crop
    res_err2 = client.get("/trends?crop=NonExistentCrop")
    assert_test(res_err2.status_code == 400, "GET /trends with invalid crop returns HTTP 400")

    # -------------------------------------------------------------------------
    # Test 3: Missing Fields in /predict
    # -------------------------------------------------------------------------
    print("\n--- Test Suite 3: Missing Required Fields in /predict ---")
    incomplete_payload = {
        "state": "Punjab",
        "season": "Rabi",
        "rainfall": 84.1,
        "temperature": 17.1,
        "area": 100.0,
        # crop missing
    }
    res = client.post("/predict", json=incomplete_payload)
    assert_test(res.status_code == 400, "POST /predict with missing 'crop' returns HTTP 400")
    assert_test("Missing required field: 'crop'" in res.get_json().get("error", ""), "Error message identifies missing field")

    # -------------------------------------------------------------------------
    # Test 4: Invalid State-Crop Pair
    # -------------------------------------------------------------------------
    print("\n--- Test Suite 4: Invalid State-Crop Combination ---")
    invalid_pair_payload = {
        "state": "Punjab",
        "crop": "Tapioca",  # Tapioca is not grown in Punjab in this dataset
        "season": "Kharif",
        "rainfall": 500.0,
        "temperature": 25.0,
        "area": 10.0,
    }
    res = client.post("/predict", json=invalid_pair_payload)
    assert_test(res.status_code == 400, "POST /predict with invalid state-crop returns HTTP 400")
    assert_test("Invalid crop 'Tapioca' for state 'Punjab'" in res.get_json().get("error", ""), "Error message lists valid crops for state")

    # -------------------------------------------------------------------------
    # Test 5: Invalid Season for a Valid State and Crop
    # -------------------------------------------------------------------------
    print("\n--- Test Suite 5: Invalid Season for State & Crop ---")
    invalid_season_payload = {
        "state": "Punjab",
        "crop": "Wheat",
        "season": "Kharif",  # Wheat in Punjab is only cultivated in Rabi
        "rainfall": 84.1,
        "temperature": 17.1,
        "area": 100.0,
    }
    res = client.post("/predict", json=invalid_season_payload)
    assert_test(res.status_code == 400, "POST /predict with invalid season returns HTTP 400")
    assert_test("Invalid season 'Kharif' for Wheat in Punjab" in res.get_json().get("error", ""), "Error message identifies invalid season")

    # -------------------------------------------------------------------------
    # Test 6: Hard Limit Exceeded
    # -------------------------------------------------------------------------
    print("\n--- Test Suite 6: Hard Limit Exceeded ---")
    hard_limit_payload = {
        "state": "Punjab",
        "crop": "Wheat",
        "season": "Rabi",
        "rainfall": 5500.0,  # Exceeds 5000 mm hard limit
        "temperature": 17.1,
        "area": 100.0,
    }
    res = client.post("/predict", json=hard_limit_payload)
    assert_test(res.status_code == 400, "POST /predict with rainfall 5500 mm returns HTTP 400")
    assert_test("hard limit" in res.get_json().get("error", ""), "Error mentions hard limit")

    # -------------------------------------------------------------------------
    # Test 7: Rainfall Inside Hard Limits but Outside Seen Range (Warning Expected)
    # -------------------------------------------------------------------------
    print("\n--- Test Suite 7: Outside Observed Range (Warning Expected, Not Error) ---")
    # Punjab Rabi observed rainfall: min 25.4, max 205.6 mm.
    # We supply 450 mm (inside [0, 5000] hard limit, but > 205.6 mm observed).
    warning_payload = {
        "state": "Punjab",
        "crop": "Wheat",
        "season": "Rabi",
        "rainfall": 450.0,
        "temperature": 17.1,
        "area": 100.0,
    }
    res = client.post("/predict", json=warning_payload)
    assert_test(res.status_code == 200, "Prediction with out-of-range rainfall returns HTTP 200")
    warn_data = res.get_json()
    assert_test(
        isinstance(warn_data.get("warnings"), list) and len(warn_data["warnings"]) > 0,
        "Response contains warnings list with items"
    )
    assert_test(
        "outside the historical observed range" in warn_data["warnings"][0],
        "Warning message mentions outside historical observed range"
    )
    print(f"      Captured Warning: \"{warn_data['warnings'][0]}\"")

    # -------------------------------------------------------------------------
    # Test 8: Valid Prediction (Punjab, Wheat, Rabi, Typical Weather, Area=100)
    # -------------------------------------------------------------------------
    print("\n--- Test Suite 8: Valid Prediction (Punjab, Wheat, Rabi) ---")
    # Punjab Rabi typical weather from metadata: rain_mean = 84.1, temp_mean = 17.1
    punjab_wheat_payload = {
        "state": "Punjab",
        "crop": "Wheat",
        "season": "Rabi",
        "rainfall": 84.1,
        "temperature": 17.1,
        "area": 100.0,
    }
    res = client.post("/predict", json=punjab_wheat_payload)
    assert_test(res.status_code == 200, "POST /predict returns HTTP 200")
    pred_data = res.get_json()
    assert_test("predicted_yield" in pred_data and pred_data["predicted_yield"] > 0, "predicted_yield is positive float")
    assert_test("total_production" in pred_data and pred_data["total_production"] > 0, "total_production is positive float")
    expected_prod = round(pred_data["predicted_yield"] * 100.0, 2)
    assert_test(
        abs(pred_data["total_production"] - expected_prod) < 0.05,
        "total_production == predicted_yield * area"
    )
    assert_test(
        isinstance(pred_data.get("warnings"), list) and len(pred_data["warnings"]) == 0,
        "No warnings generated for typical weather values"
    )

    # -------------------------------------------------------------------------
    # Test 9: /metrics Endpoint
    # -------------------------------------------------------------------------
    print("\n--- Test Suite 9: /metrics Endpoint ---")
    res_m = client.get("/metrics")
    assert_test(res_m.status_code == 200, "GET /metrics returns HTTP 200")
    m_data = res_m.get_json()
    assert_test("validation_metrics" in m_data, "Metrics contains validation_metrics object")

    # -------------------------------------------------------------------------
    # Print Test Summary and Detailed Punjab Wheat Response
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"TEST SUMMARY: {passed} PASSED, {failed} FAILED")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("DEMO RESPONSE: State=Punjab, Crop=Wheat, Season=Rabi, Weather=(84.1mm, 17.1°C), Area=100ha")
    print("=" * 80)
    print(json.dumps(pred_data, indent=2))
    print("=" * 80 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
