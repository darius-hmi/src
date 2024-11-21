import pandas as pd
import joblib
import os
import sys


sys.path.append(os.path.abspath(os.path.join('..', '..', 'data')))
from cleanAndPrepDataFunctions import apply_form_and_last3_goals, prepare_match_data_hack, prepare_match_data, inverse_one_hot_encoder, prepare_match_data_one_hot
# Load the saved model and scaler
current_dir = os.path.dirname(os.path.abspath(__file__))
processed_data_path = os.path.join(current_dir, 'processed_data.csv') 
model_path = os.path.join(current_dir, 'svm_model.joblib')
scaler_path = os.path.join(current_dir, 'scaler.joblib')
label_encoder_path = os.path.join(current_dir, 'label_encoder.joblib')
data_hack = pd.read_csv('../../data/week12.csv')
svm_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
label_encoder = joblib.load(label_encoder_path)


matches_to_predict = [
    ('Arsenal', 'Southampton'),
    ('Arsenal', 'Nott\'ham Forest'),
    ('Arsenal', 'Crystal Palace'),
    ('Arsenal', 'Brighton'),
    ('Arsenal', 'Brentford'),
    ('Arsenal', 'Wolves'),
    ('Arsenal', 'Ipswich Town'),
    ('Arsenal', 'Everton'),
    ('Arsenal', 'Manchester Utd'),
    ('Arsenal', 'West Ham')
    # Add more matches as needed
]

columns_to_update = ['Home_Form', 'Away_Form', 'Home_Form2', 'Away_Form2', 'Home_Goals_Last_3', 'Away_Goals_Last_3', 'Home_Goals_Conceded_Last_3', 'Away_Goals_Conceded_Last_3']

# Iterate over the matches
for home_team, away_team in matches_to_predict:
    data = pd.read_csv(processed_data_path)
    # Prepare the new match data
    match_data_hack = prepare_match_data_hack(data_hack, home_team, away_team)
    data_with_new_match = pd.concat([data_hack, match_data_hack], ignore_index=True)

    # Apply forms and goals
    data_with_new_match = apply_form_and_last3_goals(data_with_new_match)
    updated_match_data = data_with_new_match.iloc[-1:]

    data = inverse_one_hot_encoder(data, label_encoder)
    # Prepare the match data for prediction
    match_data = prepare_match_data_one_hot(data, home_team, away_team, label_encoder)

    match_data.to_csv('savedcheckasdads.csv')

    # Update the relevant columns with the newly computed values
    for col in columns_to_update:
        match_data.at[match_data.index[-1], col] = updated_match_data.iloc[0][col]

    # Scale the data and make predictions
    match_data_scaled = scaler.transform(match_data)
    prediction = svm_model.predict(match_data_scaled)
    prediction_proba = svm_model.predict_proba(match_data_scaled)

    # Determine the predicted class and probability
    predicted_class = prediction_proba.argmax(axis=1)
    class_labels = svm_model.classes_
    predicted_label = class_labels[predicted_class][0]
    predicted_probability = prediction_proba[0, predicted_class][0]

    # Print the prediction results for the match
    print(f"Prediction for {home_team} vs {away_team}: {predicted_label} with probability: {predicted_probability*100:.2f}%")