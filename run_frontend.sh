#!/bin/bash
# Start the Sensei AI React + Vite frontend
# Runs at http://localhost:5173 — proxies /api to http://localhost:8000

set -e
cd "$(dirname "$0")/frontend"

echo "Installing Node dependencies..."
npm install

echo "Starting Sensei AI frontend on http://localhost:5173"
npm run dev
