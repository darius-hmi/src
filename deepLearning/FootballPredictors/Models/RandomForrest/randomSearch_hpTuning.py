import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
import sys
import os

sys.path.append(os.path.abspath(os.path.join('..', '..', 'data')))
from cleanAndPrepDataFunctions import prepare_data_for_training, prepare_data_for_training_binary
# Load and prepare data

data = pd.read_csv('../../data/week12.csv')
X, y, df, label_encoder = prepare_data_for_training_binary(data)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameter search space
param_dist = {
    'n_estimators': randint(100, 2000),  # Number of trees
    'max_depth': [None, 10, 20, 30, 40, 50],  # Maximum depth of the trees
    'min_samples_split': randint(2, 20),  # Minimum samples required to split an internal node
    'min_samples_leaf': randint(1, 20),  # Minimum samples required at a leaf node
    'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider at every split
    'bootstrap': [True, False],  # Whether bootstrap samples are used when building trees
}

# Set up the RandomizedSearchCV
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=100,  # Number of random combinations to test
    cv=5,  # 5-fold cross-validation
    verbose=2,  # Print progress
    n_jobs=-1,  # Use all CPU cores
    random_state=42
)

# Fit the model
random_search.fit(X_train, y_train)

# Output best hyperparameters and score
print(f"Best Parameters: {random_search.best_params_}")
print(f"Best Cross-validation Score: {random_search.best_score_}")

# Get the best estimator
best_rf_model = random_search.best_estimator_

# Evaluate on training data
train_score = best_rf_model.score(X_train, y_train)
print(f"Train Accuracy: {train_score}")

# Evaluate on test data
test_score = best_rf_model.score(X_test, y_test)
print(f"Test Accuracy: {test_score}")
