import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import sys
import os
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join('..', '..', 'data')))
from cleanAndPrepDataFunctions import prepare_data_for_training

# Load and prepare data
data = pd.read_csv('../../data/week12.csv')
X, y, df, label_encoder = prepare_data_for_training(data)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline for scaling and PCA
pca_pipeline = make_pipeline(StandardScaler(), PCA())

# Apply the pipeline to both training and test data
X_train_rotated = pca_pipeline.fit_transform(X_train)
X_test_rotated = pca_pipeline.transform(X_test)

# Train the Decision Tree Classifier
tree_clf_pca = DecisionTreeClassifier(max_depth=10, min_samples_leaf=9, max_leaf_nodes=8, min_samples_split=4, min_weight_fraction_leaf=0.155, random_state=42)
tree_clf_pca.fit(X_train_rotated, y_train)

# Evaluate the model
train_accuracy = tree_clf_pca.score(X_train_rotated, y_train)
test_accuracy = tree_clf_pca.score(X_test_rotated, y_test)

print(f"Train Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")
