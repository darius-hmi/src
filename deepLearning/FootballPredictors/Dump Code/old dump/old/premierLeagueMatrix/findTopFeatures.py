import pandas as pd
import numpy as np

# Load dataset
match_results = pd.read_csv('data/combined_fixtures_with_results.csv')

# Get a summary of the data
print(match_results.info())
print(match_results.describe())

from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Define features and target
features = match_results.drop(columns=['Result', 'Score'])  # Exclude target
target = match_results['Result']

# Handle categorical data (if any)
features = pd.get_dummies(features)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(features, target)

# Get feature importances
importances = model.feature_importances_

# Sort and plot the top 20 important features
indices = np.argsort(importances)[-20:]  # Top 20 features
plt.figure(figsize=(10, 8))
plt.title('Top 20 Feature Importances')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [features.columns[i] for i in indices])
plt.show()
