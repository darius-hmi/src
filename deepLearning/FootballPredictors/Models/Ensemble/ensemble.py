import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import sys
import os
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import joblib

sys.path.append(os.path.abspath(os.path.join('..', '..', 'data')))
from cleanAndPrepDataFunctions import prepare_data_for_training, prepare_data_for_training_binary

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model.joblib')
scaler_path = os.path.join(current_dir, 'scaler.joblib')
processed_data_path = os.path.join(current_dir, 'processed_data.csv')
label_encoder_path = os.path.join(current_dir, 'label_encoder.joblib')


# Load and prepare data
data = pd.read_csv('../../data/week13New.csv')
X, y, df, label_encoder = prepare_data_for_training_binary(data)

df.to_csv(processed_data_path, index=False)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize StandardScaler
scaler = StandardScaler()

# Scale the data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define your models
log_reg = LogisticRegression(C=0.1, max_iter=30, solver='saga', random_state=42)
svm = SVC(kernel='rbf', C=10000, gamma=0.000001, probability=True, random_state=42)
neural_net = MLPClassifier(
    hidden_layer_sizes=(500),  # Two hidden layers, 100 and 50 neurons
    activation='logistic',             # ReLU activation
    solver='adam',                 # Adam optimizer
    alpha=0.7,                   # Regularization strength
    learning_rate='adaptive',      # Adaptive learning rate
    learning_rate_init=0.001,      # Initial learning rate
    max_iter=2000,                 # Maximum iterations
    batch_size='auto',             # Auto batch size
    random_state=42,               # Fixed random state
    tol=1e-4,                      # Tolerance for optimization
    early_stopping=True,           # Enable early stopping
    validation_fraction=0.2,       # 20% of data used for validation
    n_iter_no_change=10           # Stop after 10 iterations without improvement
)


# Create the Voting Classifier
voting_clf = VotingClassifier(
    estimators=[
        ('lr', log_reg),
        ('svm', svm),
        ('nn', neural_net)
    ],
    voting='soft'  # Use 'hard' for majority voting
)

# Train and evaluate the ensemble model
voting_clf.fit(X_train_scaled, y_train)
print("Ensemble Train Accuracy:", voting_clf.score(X_train_scaled, y_train))
print("Ensemble Test Accuracy:", voting_clf.score(X_test_scaled, y_test))


joblib.dump(voting_clf, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(label_encoder, label_encoder_path)

# Evaluate individual models
models = [log_reg, svm, neural_net]

for model in models:
    model.fit(X_train_scaled, y_train)
    print(f"{model.__class__.__name__} Train Accuracy: {model.score(X_train_scaled, y_train):.2f}")
    print(f"{model.__class__.__name__} Test Accuracy: {model.score(X_test_scaled, y_test):.2f}")
    print("\n")


# Ensemble Train Accuracy: 0.8899253731343284
# Ensemble Test Accuracy: 0.7910447761194029
# LogisticRegression Train Accuracy: 0.83
# LogisticRegression Test Accuracy: 0.79


# SVC Train Accuracy: 0.84
# SVC Test Accuracy: 0.77


# MLPClassifier Train Accuracy: 0.93
# MLPClassifier Test Accuracy: 0.77