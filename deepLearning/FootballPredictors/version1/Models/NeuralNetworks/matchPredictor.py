import pandas as pd
import joblib
import os
import sys


sys.path.append(os.path.abspath(os.path.join('..', '..', 'data')))
from cleanAndPrepDataFunctions import apply_form_and_last3_goals, prepare_match_data_hack, prepare_match_data
# Load the saved model and scaler
current_dir = os.path.dirname(os.path.abspath(__file__))
processed_data_path = os.path.join(current_dir, 'processed_data.csv') 
model_path = os.path.join(current_dir, 'nn_model.joblib')
scaler_path = os.path.join(current_dir, 'scaler.joblib')
label_encoder_path = os.path.join(current_dir, 'label_encoder.joblib')
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
label_encoder = joblib.load(label_encoder_path)


matches_to_predict = [
    # ('Newcastle Utd', 'West Ham'),
    # ('Ipswich Town', 'Manchester Utd'),
    # ('Southampton', 'Liverpool'),
    # ('Manchester City', 'Tottenham'),
    # ('Arsenal', 'Nott\'ham Forest'),
    # ('Aston Villa', 'Crystal Palace'),
    # ('Bournemouth', 'Brighton'),
    # ('Brentford', 'Everton'),
    # ('Fulham', 'Wolves'),
    # ('Leicester City', 'Chelsea')
    
    ('Everton', 'Liverpool'),
    ('Aston Villa', 'Southampton'),
    ('Brentford', 'Newcastle Utd'),
    ('Crystal Palace', 'Manchester City'),
    ('Manchester Utd', 'Nott\'ham Forest'),
    ('Aston Villa', 'Brentford'),
    ('Fulham', 'Arsenal'),
    ('Ipswich Town', 'Bournemouth'),
    ('Leicester City', 'Brighton'),
    ('Tottenham', 'Chelsea'),
    ('West Ham', 'Wolves'),
]
# remove the columns below and remove line containing apply_form_and_last3_goals from the for loop below to make the stats more produciton friendly
columns_to_update = ['Home_Form', 'Away_Form', 'Home_Form2', 'Away_Form2', 'Home_Goals_Last_3', 'Away_Goals_Last_3', 'Home_Goals_Conceded_Last_3', 'Away_Goals_Conceded_Last_3']

# Iterate over the matches
for home_team, away_team in matches_to_predict:
    data_hack = pd.read_csv('../../data/week14.csv')
    #data_hack = data_hack.drop(columns=['Home_Form', 'Away_Form', 'Home_Form2', 'Away_Form2', 'Home_Goals_Last_3', 'Away_Goals_Last_3', 'Home_Goals_Conceded_Last_3', 'Away_Goals_Conceded_Last_3'])
    data = pd.read_csv(processed_data_path)

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

    match_data_scaled = scaler.transform(match_data)
    probabilities = model.predict(match_data_scaled)[0]

    home_win_prob = probabilities[1]
    draw_prob = probabilities[0]
    away_win_prob = probabilities[2]
    # Find the predicted class (the class with the highest probability)

    # Print the prediction results for the match
    # Print the prediction results for the match, formatted as requested

    #, A:{prob_away_win * 100:.0f}
    print(f"{home_team} vs {away_team} - H:{home_win_prob * 100:.0f}, A or D:{draw_prob * 100:.0f}, A:{away_win_prob * 100:.0f}")
