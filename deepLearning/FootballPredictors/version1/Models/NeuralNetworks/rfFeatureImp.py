from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join('..', '..', 'data')))
from cleanAndPrepDataFunctions import prepare_data_for_training, prepare_data_for_training_binary

# Load and prepare data
data = pd.read_csv('../../data/week13.csv')
X, y, df, label_encoder = prepare_data_for_training(data)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit a RandomForestClassifier to your data
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# Get feature importances
feature_importances = rf.feature_importances_

# Get the indices of the 200 least important features
least_important_features_idx = np.argsort(feature_importances)[:200]

# Get the names of the 200 least important features (if available)
least_important_features = X.columns[least_important_features_idx]

current_dir = os.path.dirname(os.path.abspath(__file__))
# Save column names to a file
column_names_path = os.path.join(current_dir, 'column_names.txt')
with open(column_names_path, 'w') as f:
    for col in least_important_features:
        f.write(col + '\n')

print(f"Column names saved to {column_names_path}")

data2 = pd.read_csv('../../data/week13.csv')

columns_to_remove = list(set(least_important_features))

# Ensure both "Home_" and "Away_" versions of the columns are removed if one exists
columns_to_remove_final = []
for col in columns_to_remove:
    if col.startswith('Home_'):
        # Remove the corresponding Away column if it exists
        corresponding_away_col = 'Away_' + col[5:]
        if corresponding_away_col not in columns_to_remove_final:
            columns_to_remove_final.append(corresponding_away_col)
    if col.startswith('Away_'):
        # Remove the corresponding Home column if it exists
        corresponding_home_col = 'Home_' + col[5:]
        if corresponding_home_col not in columns_to_remove_final:
            columns_to_remove_final.append(corresponding_home_col)
    if col not in columns_to_remove_final:
        columns_to_remove_final.append(col)

# Read the data
# Remove the columns listed in columns_to_remove_final
data4 = data2.drop(columns=columns_to_remove_final, errors='ignore')

# Apply the function to drop the columns
data4.to_csv('fianlForReal.csv')

# Print the least important features
print(f"Least important features: {least_important_features}")
