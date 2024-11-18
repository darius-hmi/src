import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import xgboost as xgb

# Load the dataset
df = pd.read_csv('data.csv')

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Encode categorical features
df['Home'] = label_encoder.fit_transform(df['Home'])
df['Away'] = label_encoder.fit_transform(df['Away'])
df['Season'] = label_encoder.fit_transform(df['Season'])

# Encode target variable
y = label_encoder.fit_transform(df['Result'])

# Drop the target variable to create the feature matrix
X = df.drop(columns=['Result'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost Classifier
model = XGBClassifier()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Plot feature importance
plt.figure(figsize=(10, 8))
xgb.plot_importance(model, importance_type='weight', max_num_features=10, title='Feature Importance')
plt.show()






# import pandas as pd
# from xgboost import XGBClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.preprocessing import LabelEncoder
# import matplotlib.pyplot as plt
# import xgboost as xgb

# # Load and preprocess data
# df = pd.read_csv('data.csv')

# # Separate the target variable and encode it
# y = df['Result']
# target_encoder = LabelEncoder()
# y_encoded = target_encoder.fit_transform(y)

# # One-hot encode categorical features
# df_encoded = pd.get_dummies(df, columns=['Home', 'Away', 'Wk'])

# # Ensure all columns are numeric
# df_encoded = df_encoded.apply(pd.to_numeric, errors='ignore')

# # Add the encoded target variable
# df_encoded['Result'] = y_encoded

# # Define features and target
# X = df_encoded.drop(columns=['Result', 'Season'])
# y = df_encoded['Result']

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize and train XGBoost Classifier
# model = XGBClassifier(scale_pos_weight=1)
# model.fit(X_train, y_train)

# # Evaluate the model on training data
# y_train_pred = model.predict(X_train)
# print("\nTraining Accuracy:")
# print(f"Accuracy: {accuracy_score(y_train, y_train_pred)}")
# print(classification_report(y_train, y_train_pred, target_names=target_encoder.classes_))

# # Evaluate the model on test data
# y_pred = model.predict(X_test)
# print("\nTest Accuracy:")
# print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
# print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

# # Plot feature importance
# plt.figure(figsize=(10, 8))
# xgb.plot_importance(model, importance_type='weight', max_num_features=20, title='Feature Importance')
# plt.show()

# # Function to predict match result
# def predict_match_result(home_team, away_team, week):
#     # Create input data
#     input_data = pd.DataFrame({
#         'Home': [home_team],
#         'Away': [away_team],
#         'Wk': [week]
#     })
    
#     # One-hot encode the input data
#     input_data_encoded = pd.get_dummies(input_data, columns=['Home', 'Away', 'Wk'])
    
#     # Ensure input data has the same columns as training data
#     input_data_encoded = input_data_encoded.reindex(columns=X.columns, fill_value=0)
    
#     # Print encoded input data to debug
#     print("Encoded input data:")
#     print(input_data_encoded)
    
#     # Predict and decode the result
#     prediction_encoded = model.predict(input_data_encoded)
#     print("Encoded prediction:", prediction_encoded)  # Debugging step
#     prediction = target_encoder.inverse_transform(prediction_encoded)
#     print("Decoded prediction:", prediction)  # Debugging step
    
#     return prediction[0]

# # Example usage
# home_team = 'Brentford'
# away_team = 'Man City'
# week = '4'

# result = predict_match_result(home_team, away_team, week)
# print(f'\nThe predicted result for the match is: {result}')






# Above uses a one-hot encoding and below uses encoding categorical features



# import pandas as pd
# from xgboost import XGBClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.preprocessing import LabelEncoder
# import matplotlib.pyplot as plt
# import xgboost as xgb

# # Load the dataset
# df = pd.read_csv('data.csv')

# # Initialize the LabelEncoder
# label_encoder = LabelEncoder()

# # Encode categorical features
# df['Home'] = label_encoder.fit_transform(df['Home'])
# df['Away'] = label_encoder.fit_transform(df['Away'])
# df['Season'] = label_encoder.fit_transform(df['Season'])

# # Encode target variable
# y = label_encoder.fit_transform(df['Result'])

# # Drop the target variable to create the feature matrix
# X = df.drop(columns=['Result'])

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize XGBoost Classifier
# model = XGBClassifier()

# # Train the model
# model.fit(X_train, y_train)

# # Make predictions
# y_pred = model.predict(X_test)

# # Evaluate the model
# print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
# print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# # Plot feature importance
# plt.figure(figsize=(10, 8))
# xgb.plot_importance(model, importance_type='weight', max_num_features=10, title='Feature Importance')
# plt.show()
