#!/usr/bin/env bash
# Example calls with curl. Run from the repository root with the service up (docker compose up).
set -euo pipefail
API="${API_URL:-http://localhost:8000}"
D=data/2nd_test

echo "== health"
curl -s "$API/health"; echo

echo "== train with the default parameters (body optional)"
curl -s -X POST "$API/train" -H "Content-Type: application/json" \
     -d '{"dataset": "2nd_test", "healthy_hours": 48, "quantile": 0.995, "persistence": 3}' | head -c 1200; echo; echo

echo "== predict: upload raw IMS files (healthy, then around the B1 onset)"
curl -s -X POST "$API/predict/files" \
     -F "files=@$D/2004.02.14.10.32.39" \
     -F "files=@$D/2004.02.16.03.22.39" \
     -F "files=@$D/2004.02.16.03.32.39" \
     -F "files=@$D/2004.02.16.03.42.39" \
  | python3 -c 'import json,sys; r=json.load(sys.stdin); [print(x["id"], "anomalous:", x["anomalous_bearings"], "| B1 score", x["bearings"]["B1"]["score"], "alarm", x["bearings"]["B1"]["alarm"]) for x in r["results"]]'

echo "== predict: JSON body (built from a file by the example client)"
python3 examples/client.py payload "$D/2004.02.18.12.02.39" -o payload.json
curl -s -X POST "$API/predict" -H "Content-Type: application/json" -d @payload.json | head -c 600; echo; echo

echo "== errors: invalid input -> 422"
curl -s -X POST "$API/predict" -H "Content-Type: application/json" \
     -d '{"acquisitions": [{"signals": {"B1": [0.1, 0.2, 0.3]}}]}'; echo

echo "== models"
curl -s "$API/models"; echo
