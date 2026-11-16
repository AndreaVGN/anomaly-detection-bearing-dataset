#!/usr/bin/env bash
# Creates the virtual environment .venv and installs all dependencies (service, notebook, tests).
# Usage: ./setup.sh            (uses python3)
#        PYTHON=python3.12 ./setup.sh
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
    echo "Python >= 3.11 is required (found: $("$PY" --version 2>&1))."
    echo "Install a newer Python (e.g. 'brew install python@3.12') and run: PYTHON=python3.12 ./setup.sh"
    exit 1
fi

"$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-dev.txt
mkdir -p data models

echo
echo "Done. Activate the environment with:  source .venv/bin/activate"
echo "Dataset expected in:                  data/2nd_test/"
