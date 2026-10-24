import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app import app

client = app.test_client()

# 1. Test index
r = client.get('/')
assert r.status_code == 200, f'GET / failed: {r.status_code}'
assert b'AgriPredict AI' in r.data, 'Index content mismatch'
print('[PASS] GET / -> 200 OK (index.html served)')

# 2. Test style.css
r = client.get('/style.css')
assert r.status_code == 200, f'GET /style.css failed: {r.status_code}'
assert 'css' in r.content_type, f'Wrong content type: {r.content_type}'
print('[PASS] GET /style.css -> 200 OK')

# 3. Test script.js
r = client.get('/script.js')
assert r.status_code == 200, f'GET /script.js failed: {r.status_code}'
assert 'javascript' in r.content_type, f'Wrong content type: {r.content_type}'
print('[PASS] GET /script.js -> 200 OK')

# 4. Test API endpoints taking priority
r = client.get('/health')
assert r.status_code == 200 and r.get_json() == {'status': 'ok'}
print('[PASS] GET /health -> 200 OK')

r = client.get('/options')
assert r.status_code == 200 and 'crops' in r.get_json()
crops = r.get_json()['crops']
print(f'[PASS] GET /options -> 200 OK ({len(crops)} crops)')

r = client.get('/metrics')
assert r.status_code == 200 and 'model_name' in r.get_json()
print('[PASS] GET /metrics -> 200 OK')

r = client.get('/trends?crop=Rice')
assert r.status_code == 200 and 'trends' in r.get_json()
print('[PASS] GET /trends?crop=Rice -> 200 OK')

# 5. Test 404 on unknown file
r = client.get('/unknown_file_that_does_not_exist.xyz')
assert r.status_code == 404, f'Expected 404, got {r.status_code}'
print('[PASS] GET /unknown_file... -> 404 Not Found')

print('ALL ENDPOINT TESTS PASSED!')
