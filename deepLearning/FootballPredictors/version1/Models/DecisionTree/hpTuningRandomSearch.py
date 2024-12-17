from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'svm_model.joblib')
scaler_path = os.path.join(current_dir, 'scaler.joblib')
processed_data_path = os.path.join(current_dir, 'processed_data.csv')
label_encoder_path = os.path.join(current_dir, 'label_encoder.joblib')

sys.path.append(os.path.abspath(os.path.join('..', '..', 'data')))
from cleanAndPrepDataFunctions import prepare_data_for_training, prepare_data_for_training_binary

data = pd.read_csv('../../data/week12.csv')
X, y, df, label_encoder = prepare_data_for_training_binary(data)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_dist = {
    'max_features': [None, 'sqrt', 'log2'],
    'max_depth': randint(3, 10),
    'max_leaf_nodes': randint(5, 30),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 10),
    'min_weight_fraction_leaf': [0.0, 0.05, 0.1, 0.3]
}

tree_clf = DecisionTreeClassifier(random_state=42)

random_search = RandomizedSearchCV(
    estimator=tree_clf,
    param_distributions=param_dist,
    n_iter=100,  # Number of parameter settings sampled
    scoring='accuracy',
    cv=5,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

# Get all results as a DataFrame for easy analysis
# Convert results to a DataFrame for easier analysis
results_df = pd.DataFrame(random_search.cv_results_)

# Sort by mean test score in descending order
sorted_results = results_df.sort_values(by="mean_test_score", ascending=False)

# Display the top 5 models
top_n = 5  # Adjust to display more or fewer models
for i in range(top_n):
    print(f"Model Rank: {i + 1}")
    print(f"Mean CV Accuracy: {sorted_results.iloc[i]['mean_test_score']:.4f}")
    print(f"Parameters: {sorted_results.iloc[i]['params']}\n")



