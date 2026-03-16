#!/bin/bash
API_PID=""
FRONTEND_PID=""

cleanup() {
  echo ""
  echo "Shutting down..."
  [ -n "$API_PID" ] && kill "$API_PID" 2>/dev/null
  [ -n "$FRONTEND_PID" ] && kill "$FRONTEND_PID" 2>/dev/null
  wait "$API_PID" "$FRONTEND_PID" 2>/dev/null
  exit 0
}
trap cleanup SIGINT SIGTERM

# Start FastAPI backend
uvicorn api.main:app --reload --port 8000 &
API_PID=$!

# Start React frontend dev server
cd frontend && npm run dev &
FRONTEND_PID=$!

wait
