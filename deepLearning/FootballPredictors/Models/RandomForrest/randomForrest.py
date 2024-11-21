import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import sys
import os
from sklearn.ensemble import RandomForestClassifier

sys.path.append(os.path.abspath(os.path.join('..', '..', 'data')))
from cleanAndPrepDataFunctions import prepare_data_for_training, prepare_data_for_training_binary

# Load and prepare data
data = pd.read_csv('../../data/week12.csv')
X, y, df, label_encoder = prepare_data_for_training_binary(data)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rnd_clf = RandomForestClassifier(n_estimators=1000, max_leaf_nodes=24,
                                 n_jobs=-1, random_state=42)

rnd_clf.fit(X_train, y_train)
print(rnd_clf.score(X_train, y_train))
print(rnd_clf.score(X_test, y_test))

