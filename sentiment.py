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

embedding_dim=int(sys.argv[1])
epochs=int(sys.argv[2])
maxlen=int(sys.argv[3])
tracking_uri = sys.argv[4]

num_words=25000

data = pd.read_csv('amazon-reviews.csv', encoding="latin-1")
le = LabelEncoder()
data['class'] = le.fit_transform(data.label)
X_train, X_test, y_train, y_test = train_test_split(data.text, data['class'])


# Tokenization and padding
tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

sequences = tokenizer.texts_to_sequences(X_train)
padded = pad_sequences(sequences, maxlen=maxlen, truncating="post", padding="post")

test_sequences = tokenizer.texts_to_sequences(X_test)
test_padded = pad_sequences(test_sequences, maxlen=maxlen, truncating="post", padding="post")


mlflow.set_tracking_uri(tracking_uri)
mlflow.end_run()

with mlflow.start_run():

    # Model training
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(num_words,embedding_dim,input_length=maxlen),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    model.summary()
    

    history = model.fit(padded, y_train, epochs=epochs, validation_data=(test_padded, y_test))

    metrics = pd.DataFrame(history.history).iloc[-1,:]


    mlflow.log_param("sentence length", maxlen)
    mlflow.log_param("embedding_dim", embedding_dim)
    mlflow.log_param("num_words", num_words)
    mlflow.log_param("epochs", epochs)

    mlflow.log_metric("accuracy", metrics["accuracy"])
    mlflow.log_metric("val_accuracy", metrics["val_accuracy"])

    joblib.dump(tokenizer,"./tokenizer.tok")
    mlflow.log_artifact("./tokenizer.tok")


    mlflow.keras.log_model(model, "sentiment-analyzer")