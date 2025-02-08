#!/bin/bash

# Activate virtual environment (if using one)
source activate MLOPS 2>/dev/null

# Run FastAPI server in the background
uvicorn main:app --host 127.0.0.1 --port 8000 --reload > fastapi.log 2>&1 &

# Run MLflow UI
mlflow ui

# Print running processes
echo "FastAPI and MLflow UI are running..."