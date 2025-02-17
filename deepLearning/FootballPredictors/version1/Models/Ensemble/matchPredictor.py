import pandas as pd
import joblib
import os
import sys

sys.path.append(os.path.abspath(os.path.join('..', '..', 'data')))
from cleanAndPrepDataFunctions import apply_form_and_last3_goals, prepare_match_data_hack, prepare_match_data

# Load the saved model and scaler
current_dir = os.path.dirname(os.path.abspath(__file__))
processed_data_path = os.path.join(current_dir, 'processed_data.csv')
data = pd.read_csv(processed_data_path)
model_path = os.path.join(current_dir, 'model.joblib')
scaler_path = os.path.join(current_dir, 'scaler.joblib')
label_encoder_path = os.path.join(current_dir, 'label_encoder.joblib')
data_hack = pd.read_csv('../../data/week13New.csv')

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
label_encoder = joblib.load(label_encoder_path)

matches_to_predict = [
    ('Leicester City', 'Chelsea'),
    ('Arsenal', 'Nott\'ham Forest'),
    ('Aston Villa', 'Crystal Palace'),
    ('Bournemouth', 'Brighton'),
    ('Everton', 'Brentford'),
    ('Fulham', 'Wolves'),
    ('Manchester City', 'Tottenham'),
    ('Southampton', 'Liverpool'),
    ('Ipswich Town', 'Manchester Utd'),
    ('Newcastle Utd', 'West Ham')
]

columns_to_update = [
    'Home_Form', 'Away_Form', 'Home_Form2', 'Away_Form2', 
    'Home_Goals_Last_3', 'Away_Goals_Last_3', 'Home_Goals_Conceded_Last_3', 'Away_Goals_Conceded_Last_3'
]

# Iterate over the matches
for home_team, away_team in matches_to_predict:
    # Prepare the new match data
    match_data_hack = prepare_match_data_hack(data_hack, home_team, away_team)
    data_with_new_match = pd.concat([data_hack, match_data_hack], ignore_index=True)

    # Apply forms and goals
    data_with_new_match = apply_form_and_last3_goals(data_with_new_match)
    updated_match_data = data_with_new_match.iloc[-1:]

    # Prepare the match data for prediction
    match_data = prepare_match_data(data, home_team, away_team, label_encoder)

    # Update the relevant columns with the newly computed values
    for col in columns_to_update:
        match_data.at[match_data.index[-1], col] = updated_match_data.iloc[0][col]

    # Scale the data and make predictions
    match_data_scaled = scaler.transform(match_data)
    prediction = model.predict(match_data_scaled)

    # Assuming model outputs a single prediction per match, which is a label
    predicted_label = prediction[0]

    # If your model outputs probabilities and you want to include them:
    prediction_proba = model.predict_proba(match_data_scaled)
    predicted_probability = prediction_proba[0].max()

    # Print the prediction results for the match
    print(f"Prediction for {home_team} vs {away_team}: {predicted_label} with probability: {predicted_probability*100:.2f}%")
