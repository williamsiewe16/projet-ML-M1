#!/usr/bin/env bash

echo "Deploying Production model name=SentimentAnalyzerModel"

# Set enviroment variable for the tracking URL where the Model Registry is
export MLFLOW_TRACKING_URI=http://localhost:5000


endpoint_name='sentiment-analyzer-api'
model_uri="models:/SentimentAnalyzerModel/production"
image_url="662182664068.dkr.ecr.eu-west-3.amazonaws.com/mlflow-pyfunc:1.25.1"
role="arn:aws:iam::662182664068:role/AmazonSageMaker-ExecutionRole-20220505T133048"
region_name='eu-west-3'
instance_type='ml.m5.xlarge'
instance_count=1
mode='replace'


mlflow sagemaker deploy \
    --app-name $endpoint_name \
    --model-uri $model_uri \
    --image-url $image_url \
    --execution-role-arn $role \
    --region-name $region_name \
    --instance-type $instance_type \
    --instance-count $instance_count \
    --mode $mode