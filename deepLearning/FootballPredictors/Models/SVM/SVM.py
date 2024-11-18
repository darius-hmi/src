import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'svm_model.joblib')
scaler_path = os.path.join(current_dir, 'scaler.joblib')
processed_data_path = os.path.join(current_dir, 'processed_data.csv')
label_encoder_path = os.path.join(current_dir, 'label_encoder.joblib')

sys.path.append(os.path.abspath(os.path.join('..', '..', 'data')))
from cleanAndPrepDataFunctions import apply_one_hot_encoder, drop_seaon_col, apply_scoreToResult_01minus1, apply_scoreToResult_012, apply_label_encoder, order_features_and_prepare_target

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

svm_model = SVC(kernel='rbf', C=10000, gamma=0.000001 , random_state=42)
svm_model.fit(X_train_scaled, y_train)

joblib.dump(svm_model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(label_encoder, label_encoder_path)

y_train_pred = svm_model.predict(X_train_scaled)

y_pred = svm_model.predict(X_test_scaled)

# Evaluate the model on the training data
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"SVM Model - Training Accuracy: {train_accuracy}")
print("Training Classification Report:")
print(classification_report(y_train, y_train_pred))

# Evaluate the model on the test data
test_accuracy = accuracy_score(y_test, y_pred)
print(f"SVM Model - Test Accuracy: {test_accuracy}")
print("Test Classification Report:")
print(classification_report(y_test, y_pred))

