import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree_clf = DecisionTreeClassifier(max_depth=12, min_samples_leaf=13, random_state=42)

tree_clf.fit(X_train, y_train)
print(tree_clf.score(X_train, y_train))
print(tree_clf.score(X_test, y_test))

#tree_clf = DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, random_state=42) 61.4 %