import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import sys
import os

# Setup paths for saving models and data (you can modify this for your use case)
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'logreg_model.joblib')
scaler_path = os.path.join(current_dir, 'scaler.joblib')
processed_data_path = os.path.join(current_dir, 'processed_data.csv')
label_encoder_path = os.path.join(current_dir, 'label_encoder.joblib')

# Import custom functions (update this path based on your file structure)
sys.path.append(os.path.abspath(os.path.join('..', '..', 'data')))
from cleanAndPrepDataFunctions import prepare_data_for_training, prepare_data_for_training_binary

# Load and prepare data
data = pd.read_csv('../../data/week12.csv')
X, y, df, label_encoder = prepare_data_for_training_binary(data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define Logistic Regression model with specified hyperparameters
logreg_model = LogisticRegression(C=0.1, max_iter=30, solver='saga', random_state=42)

# Train the model
logreg_model.fit(X_train_scaled, y_train)

# Predict on the training and test data
y_train_pred = logreg_model.predict(X_train_scaled)
y_test_pred = logreg_model.predict(X_test_scaled)

# Evaluate the model on the training data
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Logistic Regression Model - Training Accuracy: {train_accuracy:.2f}")
print("Training Classification Report:")
print(classification_report(y_train, y_train_pred))

# Evaluate the model on the test data
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Logistic Regression Model - Test Accuracy: {test_accuracy:.2f}")
print("Test Classification Report:")
print(classification_report(y_test, y_test_pred))

# Save the model and scaler