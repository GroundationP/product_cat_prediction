#!/bin/bash

# Run FastAPI server in the background
echo "Starting FastAPI..."
#uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload > fastapi.log 2>&1 &
uvicorn main:app --host 0.0.0.0 --port 8000 --reload > fastapi.log 2>&1 &

sleep 2
echo "FastAPI status:"
ps aux | grep uvicorn



# Run MLflow UI
echo "Starting MLflow..."
mlflow ui --host 0.0.0.0 --port 5001