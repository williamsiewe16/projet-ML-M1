from lib2to3.pgen2 import token
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import boto3
import json
import mlflow
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os


def process(sentences,tokenizer, maxlen):
    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, maxlen=maxlen, truncating="post", padding="post")
    return padded


def predict(data, tokenizer, maxlen):
    endpoint_name = "sentiment-analyzer-api"
    data = pd.DataFrame(data=process(data, tokenizer, maxlen))

    runtime= boto3.client('runtime.sagemaker')

    # predict on the first row of the dataset
    payload = data.to_json(orient="split")

    runtime_response = runtime.invoke_endpoint(EndpointName=endpoint_name, ContentType='application/json', Body=payload)
    result = json.loads(runtime_response['Body'].read().decode())[0]['0']

    color = "lime" if result > .5 else "red"
    print(color, result)
    return result


def get_predict_params(dest):
    model_uri = "models:/SentimentAnalyzerModel/production"
    tracking_uri = "http://localhost:5000"

    mlflow.set_tracking_uri(tracking_uri)
    run = mlflow.get_run('52ad95415bf043c09402d2cfa2e1eaa7')
    mlflow.tracking.MlflowClient().download_artifacts(run.info.run_id,"tokenizer.tok",dest)

    tokenizer = joblib.load(f"{dest}/tokenizer.tok")
    return tokenizer,150





def st_space(num=1):
    for i in range(num):
        st.write("")    


def main(color="white"):
    st.title("Analyz'IT")
    with st.form("my-form"):
        text = st.text_input('Enter some text', '')
        dest = f"{os.getcwd()}/downloads"

        tokenizer = ""
        maxlen=0

        if not os.path.exists(dest):
            os.makedirs(dest)
            tokenizer, maxlen = get_predict_params(dest)
        else:
            tokenizer, maxlen = joblib.load(f"{dest}/tokenizer.tok"), 150

        st.markdown(f"<p style='color: {color}'>{text}</p>",unsafe_allow_html=True)

        submit_button = st.form_submit_button(label="Submit")

        if submit_button:
            predict([text], tokenizer, maxlen)


if __name__ == '__main__':
    main()