import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

match_results = pd.read_csv('data/cleaned_file.csv')

match_results = match_results.drop(columns=['Date', 'Score', 'Home', 'Away'])

# Define features and target
features = match_results.drop(columns=['Result'])  # Exclude target
target = match_results['Result']

# Apply SelectKBest with ANOVA F-test
selector = SelectKBest(f_classif, k=20)  # Select top 20 features
selected_features = selector.fit_transform(features, target)

# Get the selected feature names
selected_feature_names = features.columns[selector.get_support()]

print(f"Selected Features: {selected_feature_names}")
