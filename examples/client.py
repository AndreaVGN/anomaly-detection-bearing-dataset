"""Minimal client for the service (standard library only).

    python examples/client.py health
    python examples/client.py train [--dataset 2nd_test] [--healthy-hours 48]
    python examples/client.py predict data/2nd_test/2004.02.16.03.22.39 data/2nd_test/2004.02.16.03.32.39 ...
    python examples/client.py payload data/2nd_test/2004.02.16.03.22.39 -o payload.json   # JSON body for /predict

`predict` sends the raw files as JSON arrays to POST /predict and prints one line per acquisition and bearing.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

BASE_URL = os.getenv("API_URL", "http://localhost:8000")
BEARINGS = ["B1", "B2", "B3", "B4"]


def call(method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE_URL + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as err:
        sys.exit(f"HTTP {err.code}: {err.read().decode()}")


def acquisition(path: Path) -> dict:
    """One IMS file -> {"id", "timestamp", "signals": {"B1": [...], ...}}"""
    columns = list(zip(*(map(float, line.split("\t")) for line in path.read_text().splitlines() if line.strip())))
    ts = datetime.strptime(path.name, "%Y.%m.%d.%H.%M.%S").isoformat()
    return {"id": path.name, "timestamp": ts, "signals": {b: list(c) for b, c in zip(BEARINGS, columns)}}


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("health")
    t = sub.add_parser("train")
    t.add_argument("--dataset", default="2nd_test")
    t.add_argument("--healthy-hours", type=float, default=48)
    pr = sub.add_parser("predict")
    pr.add_argument("files", nargs="+", type=Path)
    pl = sub.add_parser("payload")
    pl.add_argument("files", nargs="+", type=Path)
    pl.add_argument("-o", "--output", type=Path, default=Path("payload.json"))
    args = p.parse_args()

    if args.cmd == "health":
        print(json.dumps(call("GET", "/health"), indent=2))
    elif args.cmd == "train":
        r = call("POST", "/train", {"dataset": args.dataset, "healthy_hours": args.healthy_hours})
        print(f"model {r['model_version']}  healthy period {r['healthy_period']['start']} -> {r['healthy_period']['end']}")
        for b, s in r["bearings"].items():
            print(f"  {b}: threshold {s['threshold']:.2f}  anomaly onset {s['anomaly_onset']}")
    elif args.cmd == "predict":
        r = call("POST", "/predict", {"acquisitions": [acquisition(f) for f in args.files]})
        print(f"model {r['model_version']}  (alarm = {r['persistence']} consecutive anomalous acquisitions)")
        for res in r["results"]:
            for b, v in res["bearings"].items():
                print(f"  {res['id']}  {b}  score {v['score']:6.2f} / {v['threshold']:.2f}  {v['status']:<9} alarm={v['alarm']}")
    elif args.cmd == "payload":
        args.output.write_text(json.dumps({"acquisitions": [acquisition(f) for f in args.files]}))
        print(f"written {args.output}")


if __name__ == "__main__":
    main()
