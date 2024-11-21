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
model_path = os.path.join(current_dir, 'nn_model.joblib')
scaler_path = os.path.join(current_dir, 'scaler.joblib')
label_encoder_path = os.path.join(current_dir, 'label_encoder.joblib')
data_hack = pd.read_csv('../../data/week12.csv')
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
    # Add more matches as needed
]

columns_to_update = ['Home_Form', 'Away_Form', 'Home_Form2', 'Away_Form2', 'Home_Goals_Last_3', 'Away_Goals_Last_3', 'Home_Goals_Conceded_Last_3', 'Away_Goals_Conceded_Last_3']

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
    prediction_proba = model.predict(match_data_scaled)

    # Determine the predicted class and probability
    prob_draw = prediction_proba[0, 0]  # Probability of draw (class 0)
    prob_home_win = prediction_proba[0, 1]  # Probability of home win (class 1)
    prob_away_win = prediction_proba[0, 2]  # Probability of away win (class 2)

    # Find the predicted class (the class with the highest probability)
    predicted_class = prediction_proba.argmax(axis=1)[0]  # Index of the highest probability class
    predicted_label = predicted_class

    # Print the prediction results for the match
    # Print the prediction results for the match, formatted as requested
    print(f"{home_team} vs {away_team} - H:{prob_home_win * 100:.0f}, D:{prob_draw * 100:.0f}, A:{prob_away_win * 100:.0f}")
