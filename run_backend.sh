#!/bin/bash
# Start the Sensei AI FastAPI backend
# Runs at http://localhost:8000 — API docs at http://localhost:8000/docs

set -e
cd "$(dirname "$0")"

echo "Installing Python dependencies..."
pip install -r requirements.txt -q

echo "Starting Sensei AI backend on http://localhost:8000"
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
