from calendar import EPOCH
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow  as tf
import mlflow
import sys
import warnings
import joblib

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

# get arguments

X = np.arange(1,20).reshape((-1,1))
y = 2*X+1


tracking_uri = "http://localhost:5000"

mlflow.set_tracking_uri(tracking_uri)
mlflow.end_run()

with mlflow.start_run():

    # Model trainin
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(X,y)

    mlflow.log_param('a',0.8)
    mlflow.sklearn.log_model(model, "sentiment-analyzer")