from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Define the parameter grid
param_grid = {
    'max_features': [None, 'sqrt', 'log2'],
    'max_depth': [3, 5, 6, 8, 10],
    'max_leaf_nodes': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5, 10],
    'min_weight_fraction_leaf': [0.0, 0.05, 0.1, 0.3]
}


# Initialize the DecisionTreeClassifier
tree_clf = DecisionTreeClassifier(random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(
    estimator=tree_clf,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,  # 5-fold cross-validation
    verbose=1,  # Show progress
    n_jobs=-1  # Use all available processors
)

# Fit GridSearchCV on the training data
grid_search.fit(X_train, y_train)

# Print the best parameters and score
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.2f}")

# Evaluate on the test set
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print(f"Test Set Accuracy: {test_accuracy:.2f}")
