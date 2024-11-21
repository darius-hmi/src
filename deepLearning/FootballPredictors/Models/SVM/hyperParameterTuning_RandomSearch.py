import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import sys
import os
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

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


svm_model = SVC(random_state=42)

# Define the parameter grid to search
param_dist = {
    'C': np.logspace(-3, 0, 6),  # A smaller range
    'gamma': ['scale', 'auto'] + np.logspace(-3, 3, 6).tolist(),  # A smaller range
    'kernel': ['linear'],  # Use only 'rbf' kernel to reduce combinations
    'degree': [1,2],  # Only relevant for polynomial kernels
    'coef0': [1,2,3,4,5,6],  # For polynomial or sigmoid kernels, can restrict
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=svm_model, 
    param_distributions=param_dist, 
    n_iter=100,  # Number of different combinations to try
    scoring='accuracy',  # Scoring method to optimize
    cv=5,  # Cross-validation folds
    verbose=2,  # Print progress
    random_state=42,  # Reproducibility
    n_jobs=-1  # Use all available cores
)

random_search.fit(X_train_scaled, y_train)
print("Best Hyperparameters:", random_search.best_params_)
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
from sklearn.metrics import accuracy_score
print("Test Accuracy:", accuracy_score(y_test, y_pred))
