import json
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:5000"

def test_endpoint(name, url, method="GET", data=None):
    print(f"\n--- Testing {name}: {method} {url} ---")
    headers = {}
    body = None
    if data is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(data).encode("utf-8")
    
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            status = resp.status
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read()
            print(f"Status: {status} OK")
            print(f"Content-Type: {content_type}")
            if "application/json" in content_type:
                parsed = json.loads(raw.decode("utf-8"))
                print("Response JSON:")
                print(json.dumps(parsed, indent=2)[:500] + ("..." if len(json.dumps(parsed, indent=2)) > 500 else ""))
                return parsed
            else:
                text = raw.decode("utf-8", errors="ignore")
                print(f"Body length: {len(text)} chars")
                print("First 200 chars:")
                print(text[:200])
                return text
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.read().decode('utf-8', errors='ignore')}")
        raise
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    print("Testing AgriPredict AI Local Server...")

    # 1. Test root "/"
    html = test_endpoint("Root (Index HTML)", f"{BASE_URL}/")
    assert "<!DOCTYPE html>" in html
    assert "AgriPredict AI" in html
    print("-> Root index.html served correctly from public/")

    # 2. Test static script.js
    js = test_endpoint("Static Script", f"{BASE_URL}/script.js")
    assert "const API_BASE_URL" in js
    assert "window.location.protocol === 'file:' || window.location.port === '8000'" in js
    print("-> script.js served correctly from public/ with updated API_BASE_URL")

    # 3. Test /health
    health = test_endpoint("Health Check", f"{BASE_URL}/health")
    assert health.get("status") == "ok"
    print("-> /health returned status ok")

    # 4. Test /options
    options = test_endpoint("Options", f"{BASE_URL}/options")
    crops = options.get("crops", [])
    seasons = options.get("seasons", [])
    soil_types = options.get("soil_types", [])
    print(f"-> /options returned {len(crops)} crops, {len(seasons)} seasons, {len(soil_types)} soil types")
    assert len(crops) > 0

    # 5. Test /predict
    test_payload = {
        "crop": "Wheat",
        "season": "Rabi",
        "rainfall": 200.0,
        "temperature": 20.0,
        "soil": "Alluvial",
        "area": 5.0
    }
    prediction = test_endpoint("Predict", f"{BASE_URL}/predict", method="POST", data=test_payload)
    print(f"-> /predict result: predicted_yield={prediction.get('predicted_yield')} t/ha, total_production={prediction.get('total_production')} tonnes")
    assert "predicted_yield" in prediction
    assert "total_production" in prediction

    print("\nALL LOCAL TESTS PASSED SUCCESSFULLY!")
