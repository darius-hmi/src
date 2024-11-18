import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import sys
import os

sys.path.append(os.path.abspath(os.path.join('..', '..', 'data')))
from cleanAndPrepDataFunctions import apply_one_hot_encoder, drop_seaon_col, apply_scoreToResult_01minus1, apply_scoreToResult_012, apply_label_encoder, order_features_and_prepare_target

data = pd.read_csv('../../data/week12.csv')
data, label_encoder = apply_label_encoder(data)
data = drop_seaon_col(data)
data = apply_scoreToResult_012(data)

X, y, df = order_features_and_prepare_target(data)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_model = SVC(kernel='rbf', class_weight='balanced', random_state=42)
param_grid = {
    'C': [1, 100, 1000, 10000, 100000, 1000000],  # Regularization parameter
    'gamma': [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 1],  # Kernel coefficient
    'kernel': ['rbf'],  # Kernel type
    'class_weight': ['balanced', None],  # Handle class imbalance
}

grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

grid_search.fit(X_train_scaled, y_train)

print(f"Best Hyperparameters: {grid_search.best_params_}")

best_svm_model = grid_search.best_estimator_
y_train_pred = best_svm_model.predict(X_train_scaled)
y_pred = best_svm_model.predict(X_test_scaled)

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

from sklearn.metrics import accuracy_score
print("Test Accuracy:", accuracy_score(y_test, y_pred))
