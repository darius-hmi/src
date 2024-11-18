import pandas as pd
import joblib
import os


# Load the saved model and scaler
current_dir = os.path.dirname(os.path.abspath(__file__))
processed_data_path = os.path.join(current_dir, 'processed_data.csv') 
data = pd.read_csv(processed_data_path)
model_path = os.path.join(current_dir, 'svm_model.joblib')
scaler_path = os.path.join(current_dir, 'scaler.joblib')
label_encoder_path = os.path.join(current_dir, 'label_encoder.joblib')

svm_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
label_encoder = joblib.load(label_encoder_path)


def get_last_match_stats(df, team, is_home):
    """
    Get the last match stats for a team, formatted to match their position (Home/Away) in the new match.
    """
    team_stats = df[(df['Home'] == team) | (df['Away'] == team)]
    
    # Get the most recent match (last row in the filtered dataframe)
    last_match_stats = team_stats.iloc[-1]
    
    # Select the stats based on the last match and align with the desired prefix
    if last_match_stats['Home'] == team:
        # Team was Home in the last match
        stats = last_match_stats[[col for col in last_match_stats.index if col.startswith('Home_')]]
        if is_home:
            # If the team is Home in the new match, keep the Home_ prefix
            return stats
        else:
            # If the team is Away in the new match, replace Home_ with Away_
            stats.index = [col.replace('Home_', 'Away_') for col in stats.index]
            return stats
    else:
        # Team was Away in the last match
        stats = last_match_stats[[col for col in last_match_stats.index if col.startswith('Away_')]]
        if is_home:
            # If the team is Home in the new match, replace Away_ with Home_
            stats.index = [col.replace('Away_', 'Home_') for col in stats.index]
            return stats
        else:
            # If the team is Away in the new match, keep the Away_ prefix
            return stats
    

def prepare_match_data(df, home_team, away_team):
    home_team_encoded = label_encoder.transform([home_team])[0]
    away_team_encoded = label_encoder.transform([away_team])[0]

    home_team_stats = get_last_match_stats(df, home_team_encoded, is_home=True)
    away_team_stats = get_last_match_stats(df, away_team_encoded, is_home=False)

    home_stats_row = home_team_stats.to_dict()
    away_stats_row = away_team_stats.to_dict()

    wk = df['Wk'].iloc[-1] + 1

    match_data = pd.DataFrame({
        'Wk': [wk],
        'Home': [home_team_encoded],
        'Away': [away_team_encoded],
        **home_stats_row,
        **away_stats_row
    })
    return match_data


home_team = 'Newcastle Utd'
away_team = 'West Ham'


match_data = prepare_match_data(data, home_team, away_team)
match_data_scaled = scaler.transform(match_data)
prediction = svm_model.predict(match_data_scaled)
print(f"Prediction for {home_team} vs {away_team}: {prediction}")
