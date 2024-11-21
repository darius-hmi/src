import pandas as pd
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import sys
import os
from tensorflow.keras import regularizers

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'nn_model.joblib')
scaler_path = os.path.join(current_dir, 'scaler.joblib')
processed_data_path = os.path.join(current_dir, 'processed_data.csv')
label_encoder_path = os.path.join(current_dir, 'label_encoder.joblib')

sys.path.append(os.path.abspath(os.path.join('..', '..', 'data')))
from cleanAndPrepDataFunctions import (
    apply_one_hot_encoder, drop_seaon_col, apply_scoreToResult_01minus1, 
    apply_scoreToResult_012, apply_label_encoder, order_features_and_prepare_target
)

data = pd.read_csv('../../data/week12.csv')
data, label_encoder = apply_label_encoder(data)
data = drop_seaon_col(data)
data = apply_scoreToResult_012(data)

X, y, df = order_features_and_prepare_target(data)
df.to_csv(processed_data_path, index=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


import keras_tuner as kt

def build_model(hp):
    n_hidden = hp.Int("n_hidden", min_value=0, max_value=8, default=2)
    n_neurons = hp.Int("n_neurons", min_value=16, max_value=256)
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2,
                             sampling="log")
    optimizer = hp.Choice("optimizer", values=["sgd", "adam"])
    if optimizer == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    for _ in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons, activation="relu"))
    model.add(tf.keras.layers.Dense(3, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
                  metrics=["accuracy"])
    return model

random_search_tuner = kt.RandomSearch(
    build_model, objective="val_accuracy", max_trials=20, overwrite=True,
    directory="my_fashion_mnist", project_name="my_rnd_search", seed=42)

random_search_tuner.search(X_train_scaled, y_train, validation_split=0.2, epochs=20)



