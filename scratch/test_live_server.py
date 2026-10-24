import json
import urllib.request
import urllib.error

def test_live_server():
    base = "http://127.0.0.1:5000"
    
    # 1. Test GET /
    with urllib.request.urlopen(f"{base}/") as res:
        assert res.status == 200
        html = res.read().decode("utf-8")
        assert "AgriPredict AI" in html
        assert "serverErrorBanner" in html
        print("[LIVE PASS] GET / returned 200 OK with index.html")

    # 2. Test GET /style.css
    with urllib.request.urlopen(f"{base}/style.css") as res:
        assert res.status == 200
        assert "css" in res.headers.get("Content-Type", "")
        print("[LIVE PASS] GET /style.css returned 200 OK")

    # 3. Test GET /script.js
    with urllib.request.urlopen(f"{base}/script.js") as res:
        assert res.status == 200
        assert "javascript" in res.headers.get("Content-Type", "")
        js = res.read().decode("utf-8")
        assert "API_BASE_URL" in js
        assert "Connecting to server..." in js
        assert "Retry now" in js
        print("[LIVE PASS] GET /script.js returned 200 OK with auto-reconnect logic")

    # 4. Test GET /options
    with urllib.request.urlopen(f"{base}/options") as res:
        assert res.status == 200
        data = json.loads(res.read().decode("utf-8"))
        assert len(data.get("crops", [])) == 40
        print(f"[LIVE PASS] GET /options returned 200 OK with {len(data['crops'])} crops")

    # 5. Test GET /metrics
    with urllib.request.urlopen(f"{base}/metrics") as res:
        assert res.status == 200
        data = json.loads(res.read().decode("utf-8"))
        assert "model_name" in data
        print(f"[LIVE PASS] GET /metrics returned 200 OK (Model: {data['model_name']})")

    # 6. Test GET /trends?crop=Rice
    with urllib.request.urlopen(f"{base}/trends?crop=Rice") as res:
        assert res.status == 200
        data = json.loads(res.read().decode("utf-8"))
        assert len(data.get("trends", [])) > 0
        print(f"[LIVE PASS] GET /trends?crop=Rice returned 200 OK with {len(data['trends'])} seasonal trend points")

    # 7. Test POST /predict
    payload = json.dumps({
        "crop": "Rice",
        "season": "Kharif",
        "rainfall": 850.0,
        "temperature": 27.5,
        "soil": "Alluvial",
        "area": 2.5
    }).encode("utf-8")
    req = urllib.request.Request(f"{base}/predict", data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as res:
            assert res.status == 200
            data = json.loads(res.read().decode("utf-8"))
            assert "predicted_yield" in data
            print(f"[LIVE PASS] POST /predict returned 200 OK (Predicted Yield: {data['predicted_yield']} t/ha)")
    except urllib.error.HTTPError as e:
        print("Predict error response:", e.read().decode("utf-8"))
        raise

    # 8. Test 404
    try:
        urllib.request.urlopen(f"{base}/nonexistent_file_abc123.xyz")
        assert False, "Expected 404"
    except urllib.error.HTTPError as e:
        assert e.code == 404
        print("[LIVE PASS] GET /nonexistent_file_abc123.xyz returned 404 Not Found")

    print("\nALL LIVE ENDPOINT TESTS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    test_live_server()
