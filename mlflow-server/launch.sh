#!/usr/bin/env bash

port=5000
echo "Launching mlflow server on Port $port..."

mlflow server  --host 0.0.0.0  --port $port  --default-artifact-root s3://mlflow-bucket-1607  --backend-store-uri sqlite:///mlruns.db