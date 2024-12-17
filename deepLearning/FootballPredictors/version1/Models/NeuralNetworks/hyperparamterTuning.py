import pandas as pd
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import sys
import os
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import numpy as np
from scikeras.wrappers import KerasClassifier

# Define file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'nn_model.joblib')
scaler_path = os.path.join(current_dir, 'scaler.joblib')
processed_data_path = os.path.join(current_dir, 'processed_data.csv')
label_encoder_path = os.path.join(current_dir, 'label_encoder.joblib')

# Import custom data preparation functions
sys.path.append(os.path.abspath(os.path.join('..', '..', 'data')))
from cleanAndPrepDataFunctions import prepare_data_for_training

# Load data
data = pd.read_csv('../../data/week13.csv')
X, y, df, label_encoder = prepare_data_for_training(data)
df.to_csv(processed_data_path, index=False)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define function to create model
def create_model(optimizer='adam', learning_rate=0.001, dropout_rate=0.3, l1=0.01, l2=0.6):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dense(800, activation='relu', kernel_initializer="he_normal", 
                              kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(800, activation='relu', kernel_initializer="he_normal", 
                              kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    
    model.compile(optimizer=optimizer, 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

# Wrap the model for scikit-learn
model = KerasClassifier(build_fn=create_model, verbose=1)

# Define hyperparameter grid for GridSearchCV
param_grid = {
    'optimizer': ['adam', 'sgd'],
    'learning_rate': [0.001, 0.01],
    'dropout_rate': [0.3, 0.5],
    'l1': [0.01, 0.1],
    'l2': [0.1, 0.6],
    'batch_size': [32, 64],
    'epochs': [50, 100]
}

# Set up GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)

# Alternatively, for RandomizedSearchCV (if the grid is large):
param_dist = {
    'optimizer': ['adam', 'sgd'],
    'learning_rate': np.logspace(-5, -1, 5),
    'dropout_rate': [0.3, 0.5],
    'l1': [0.01, 0.1, 0.2],
    'l2': [0.1, 0.3, 0.6],
    'batch_size': [32, 64],
    'epochs': [50, 100]
}

random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, n_jobs=-1, cv=3, random_state=42)

# Perform grid search (or randomized search)
grid_result = grid.fit(X_train_scaled, y_train)
# random_search_result = random_search.fit(X_train_scaled, y_train)

# Summarize the results
print(f"Best GridSearch: {grid_result.best_score_} using {grid_result.best_params_}")
# print(f"Best RandomizedSearch: {random_search_result.best_score_} using {random_search_result.best_params_}")

# Get the best model from grid search
best_model = grid_result.best_estimator_

# Evaluate the model on test data
test_loss, test_accuracy = best_model.score(X_test_scaled, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Save the best model and scaler
joblib.dump(best_model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(label_encoder, label_encoder_path)

# Predict on training and test data
y_train_pred_probs = best_model.predict(X_train_scaled)
y_train_pred = tf.argmax(y_train_pred_probs, axis=1).numpy()

y_test_pred_probs = best_model.predict(X_test_scaled)
y_test_pred = tf.argmax(y_test_pred_probs, axis=1).numpy()

# Training data evaluation
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Neural Network Model - Training Accuracy: {train_accuracy:.2f}")
print("Training Classification Report:")
print(classification_report(y_train, y_train_pred))

# Test data evaluation
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Neural Network Model - Test Accuracy: {test_accuracy:.2f}")
print("Test Classification Report:")
print(classification_report(y_test, y_test_pred))

# Optionally, save confusion matrix and probability plots
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Confusion matrix for training set
# train_cm = confusion_matrix(y_train, y_train_pred)
# test_cm = confusion_matrix(y_test, y_test_pred)

# # Plot confusion matrix
# def plot_confusion_matrix(cm, title):
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_,
#                 yticklabels=label_encoder.classes_)
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.title(title)
#     plt.show()

# plot_confusion_matrix(train_cm, "Training Set Confusion Matrix")
# plot_confusion_matrix(test_cm, "Test Set Confusion Matrix")

# # Plot probability distribution for each class
# for i, class_name in enumerate(label_encoder.classes_):
#     plt.figure(figsize=(6, 4))
#     sns.histplot(y_test_pred_probs[:, i], bins=20, kde=True, color=f"C{i}", label=class_name)
#     plt.title(f"Probability Distribution for Class '{class_name}'")
#     plt.xlabel("Predicted Probability")
#     plt.ylabel("Frequency")
#     plt.legend()
#     plt.show()
