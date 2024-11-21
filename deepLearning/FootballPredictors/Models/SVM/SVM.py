import pandas as pd
import joblib
from sklearn.svm import SVC
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
from cleanAndPrepDataFunctions import prepare_data_for_training, prepare_data_for_training_binary, prepare_data_for_training_oneHot

data = pd.read_csv('../../data/week13.csv')
X, y, df, label_encoder = prepare_data_for_training_binary(data)

df.to_csv(processed_data_path, index=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_model = SVC(kernel='rbf', C=100000, gamma=0.0000001,probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)

joblib.dump(svm_model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(label_encoder, label_encoder_path)

y_train_pred = svm_model.predict(X_train_scaled)

y_pred = svm_model.predict(X_test_scaled)

# Evaluate the model on the training data
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"SVM Model - Training Accuracy: {train_accuracy}")
print("Training Classification Report:")
print(classification_report(y_train, y_train_pred))

# Evaluate the model on the test data
test_accuracy = accuracy_score(y_test, y_pred)
print(f"SVM Model - Test Accuracy: {test_accuracy}")
print("Test Classification Report:")
print(classification_report(y_test, y_pred))

# from sklearn.calibration import calibration_curve
# import matplotlib.pyplot as plt

# prob_pos = svm_model.predict_proba(X_test_scaled)[:, 1]
# fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)

# plt.figure(figsize=(10, 10))
# plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="SVM")
# plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
# plt.xlabel("Mean predicted value")
# plt.ylabel("Fraction of positives")
# plt.title('Calibration Curve')
# plt.legend()
# plt.show()


# # Verify that we have Arsenal matches in the test set
# arsenal_test_indices = X_test.index[(X_test['Home'] == 1111) | (X_test['Away'] == 1111)]
# print(arsenal_test_indices)

# # Check the length of the test set
# print(f"Length of X_test_scaled: {len(X_test_scaled)}")

# # Ensure indices are within bounds
# valid_indices = [i for i in arsenal_test_indices if i < len(X_test_scaled)]
# print(f"Valid indices for Arsenal: {valid_indices}")

# # Prediction using valid indices
# arsenal_y_pred = svm_model.predict(X_test_scaled[valid_indices])
# arsenal_y_true = y_test.iloc[valid_indices]

# print("Classification Report for Arsenal Matches:")
# print(classification_report(arsenal_y_true, arsenal_y_pred))




# import matplotlib.pyplot as plt
# import numpy as np

# # Example: Feature importance (using coef_ for linear models)
# if hasattr(svm_model, 'coef_'):
#     feature_importance = np.abs(svm_model.coef_[0])
# else:
#     from sklearn.inspection import permutation_importance
#     result = permutation_importance(svm_model, X_test_scaled, y_test, n_repeats=10, random_state=42)
#     feature_importance = result.importances_mean

# # Get top 20 features
# top_n = 20
# sorted_idx = feature_importance.argsort()[-top_n:][::-1]

# # Plotting
# plt.figure(figsize=(10, 8))
# plt.barh(range(top_n), feature_importance[sorted_idx], align='center')
# plt.yticks(range(top_n), [X.columns[i] for i in sorted_idx])
# plt.xlabel('Feature Importance')
# plt.title('Top 20 Feature Importance')
# plt.gca().invert_yaxis()
# plt.show()



#pretty good even for classification
# Training Classification Report:
#               precision    recall  f1-score   support

#            A       0.80      0.86      0.83       531
#            D       0.68      0.49      0.57       360

#     accuracy                           0.79      1608
#    macro avg       0.77      0.75      0.75      1608
# weighted avg       0.78      0.79      0.78      1608

# SVM Model - Test Accuracy: 0.7039800995024875
# Test Classification Report:
#               precision    recall  f1-score   support

#            A       0.74      0.78      0.76       130
#            D       0.54      0.37      0.44       102
#            H       0.74      0.84      0.79       170

#     accuracy                           0.70       402
#    macro avg       0.67      0.67      0.66       402
# weighted avg       0.69      0.70      0.69       402