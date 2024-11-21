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
from cleanAndPrepDataFunctions import prepare_data_for_training, prepare_data_for_training_binary

data = pd.read_csv('../../data/week12.csv')
X, y, df, label_encoder = prepare_data_for_training_binary(data)

df.to_csv(processed_data_path, index=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_model = SVC(kernel='rbf', C=100000, gamma=0.0000001,probability=True, random_state=42)
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



#pretty good even for classification
# Training Classification Report:
#               precision    recall  f1-score   support

#            A       0.80      0.86      0.83       531
#            D       0.68      0.49      0.57       360

#     accuracy                           0.79      1608
#    macro avg       0.77      0.75      0.75      1608
# weighted avg       0.78      0.79      0.78      1608

# SVM Model - Test Accuracy: 0.7039800995024875
# Test Classification Report:
#               precision    recall  f1-score   support

#            A       0.74      0.78      0.76       130
#            D       0.54      0.37      0.44       102
#            H       0.74      0.84      0.79       170

#     accuracy                           0.70       402
#    macro avg       0.67      0.67      0.66       402
# weighted avg       0.69      0.70      0.69       402