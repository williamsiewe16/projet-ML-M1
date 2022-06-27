#!/usr/bin/env bash

echo "Deleting Model Endpoint"


endpoint_name='sentiment-analyzer-api'
region_name='eu-west-3'

mlflow sagemaker delete \
    --app-name $endpoint_name \
    --region-name $region_name