import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import sys
import os

sys.path.append(os.path.abspath(os.path.join('..', '..', 'data')))
from cleanAndPrepDataFunctions import prepare_data_for_training, prepare_data_for_training_binary

data = pd.read_csv('../../data/week12.csv')


def prepare_match_data(df, home_team, away_team):


    wk = df['Wk'].iloc[-1] + 1

    match_data = pd.DataFrame({
        'Wk': [wk],
        'Home': [home_team],
        'Away': [away_team],
    })
    match_data.to_csv('testMatch.csv', index=False)
    return match_data


home_team = 'West Ham'
away_team = 'Crystal Palace'


match_data = prepare_match_data(data, home_team, away_team)


data_with_new_match = pd.concat([data, match_data], ignore_index=True)
data_with_new_match.to_csv('comeon.csv')


team_form = {}

# Function to assign points based on match result
def get_points(result, is_home):
    if result == 'H':  # Home win
        return 3 if is_home else 0
    elif result == 'A':  # Away win
        return 0 if is_home else 3
    elif result == 'D':  # Draw
        return 1
    else:
        return 0  # In case of missing or incorrect result

# Add a new column to the DataFrame to store the form of each team
data_with_new_match['Home_Form'] = 7.5  # Initial average form
data_with_new_match['Away_Form'] = 7.5  # Initial average form

# Iterate over each row (match) in the DataFrame
for idx, row in data_with_new_match.iterrows():
    home_team = row['Home']
    away_team = row['Away']
    result = row['Result']  # Full-time result (H, A, D)

    # Initialize form for teams if they are not already in the dictionary
    if home_team not in team_form:
        team_form[home_team] = []
    if away_team not in team_form:
        team_form[away_team] = []

    # Calculate points for the current match
    home_points = get_points(result, is_home=True)
    away_points = get_points(result, is_home=False)

    # Add the current points to the team's form (rolling list of points)
    team_form[home_team].append(home_points)
    team_form[away_team].append(away_points)

    # If the team has played more than 5 games, calculate the form as the sum of the last 5 games
    if len(team_form[home_team]) > 5:
        data_with_new_match.at[idx, 'Home_Form'] = sum(team_form[home_team][-5:])
    if len(team_form[away_team]) > 5:
        data_with_new_match.at[idx, 'Away_Form'] = sum(team_form[away_team][-5:])



data_with_new_match['Home_Form2'] = 7.5  # Initial average form
data_with_new_match['Away_Form2'] = 7.5


# Initialize separate dictionaries for home and away forms
home_team_form = {}
away_team_form = {}

# Iterate over each row (match) in the DataFrame
for idx, row in data_with_new_match.iterrows():
    home_team = row['Home']
    away_team = row['Away']
    result = row['Result']  # Full-time result (H, A, D)

    # Initialize form for teams if they are not already in the dictionary
    if home_team not in home_team_form:
        home_team_form[home_team] = []
    if away_team not in away_team_form:
        away_team_form[away_team] = []

    # Calculate points for the current match
    home_points = get_points(result, is_home=True)
    away_points = get_points(result, is_home=False)

    # Add the current points to the team's home and away form (rolling list of points)
    home_team_form[home_team].append(home_points)
    away_team_form[away_team].append(away_points)

    # If the team has played more than 5 home or away games, calculate the form as the sum of the last 5 games
    if len(home_team_form[home_team]) > 5:
        data_with_new_match.at[idx, 'Home_Form2'] = sum(home_team_form[home_team][-5:])
    if len(away_team_form[away_team]) > 5:
        data_with_new_match.at[idx, 'Away_Form2'] = sum(away_team_form[away_team][-5:])




def get_goals_from_score(score):
    try:
        score = score.replace('–', '-').replace('—', '-').replace('−', '-')
        home_score, away_score = map(int, score.split('-'))
        return home_score, away_score
    except Exception as e:
        print(f"Error processing score '{score}': {e}")
        return None, None

# Initialize a dictionary to keep track of the last 3 games' goals for each team
team_goals = {}

# Example of how you might load your data
# Add new columns to store the total goals scored and conceded in the last 3 games
data_with_new_match['Home_Goals_Last_3'] = 0
data_with_new_match['Away_Goals_Last_3'] = 0
data_with_new_match['Home_Goals_Conceded_Last_3'] = 0
data_with_new_match['Away_Goals_Conceded_Last_3'] = 0

# Iterate over each row in the dataframe (match data)
for idx, row in data_with_new_match.iloc[:-1].iterrows():
    home_team = row['Home']
    away_team = row['Away']
    score = row['Score']  # Full-time score, e.g., '2-1'

    # Get the number of goals scored by both teams
    home_goals, away_goals = get_goals_from_score(score)

    # Initialize the goal list if it's the first match for the teams
    if home_team not in team_goals:
        team_goals[home_team] = {'scored': [], 'conceded': []}
    if away_team not in team_goals:
        team_goals[away_team] = {'scored': [], 'conceded': []}

    # Append the goals scored and conceded in the current match
    team_goals[home_team]['scored'].append(home_goals)
    team_goals[away_team]['scored'].append(away_goals)
    team_goals[home_team]['conceded'].append(away_goals)
    team_goals[away_team]['conceded'].append(home_goals)

    # For home team: calculate the total goals scored in the last 3 games excluding the current game
    if len(team_goals[home_team]['scored']) > 1:  # If there are at least 2 games, calculate
        data_with_new_match.at[idx, 'Home_Goals_Last_3'] = sum(team_goals[home_team]['scored'][-4:-1])  # sum last 3 excluding current game
    if len(team_goals[home_team]['conceded']) > 1:  # If there are at least 2 games, calculate
        data_with_new_match.at[idx, 'Home_Goals_Conceded_Last_3'] = sum(team_goals[home_team]['conceded'][-4:-1])  # sum last 3 excluding current game

    # For away team: calculate the total goals scored in the last 3 games excluding the current game
    if len(team_goals[away_team]['scored']) > 1:  # If there are at least 2 games, calculate
        data_with_new_match.at[idx, 'Away_Goals_Last_3'] = sum(team_goals[away_team]['scored'][-4:-1])  # sum last 3 excluding current game
    if len(team_goals[away_team]['conceded']) > 1:  # If there are at least 2 games, calculate
        data_with_new_match.at[idx, 'Away_Goals_Conceded_Last_3'] = sum(team_goals[away_team]['conceded'][-4:-1])  # sum last 3 excluding current game

last_row_idx = data_with_new_match.index[-1]
last_row = data_with_new_match.iloc[-1]
home_team = last_row['Home']
away_team = last_row['Away']

# Compute values based on the last 3 matches
if len(team_goals[home_team]['scored']) >= 3:
    data_with_new_match.at[last_row_idx, 'Home_Goals_Last_3'] = sum(team_goals[home_team]['scored'][-3:])
if len(team_goals[home_team]['conceded']) >= 3:
    data_with_new_match.at[last_row_idx, 'Home_Goals_Conceded_Last_3'] = sum(team_goals[home_team]['conceded'][-3:])

if len(team_goals[away_team]['scored']) >= 3:
    data_with_new_match.at[last_row_idx, 'Away_Goals_Last_3'] = sum(team_goals[away_team]['scored'][-3:])
if len(team_goals[away_team]['conceded']) >= 3:
    data_with_new_match.at[last_row_idx, 'Away_Goals_Conceded_Last_3'] = sum(team_goals[away_team]['conceded'][-3:])

updated_match_data = data_with_new_match.iloc[-1:]

updated_match_data.to_csv('matchEng.csv')

final = pd.concat([data, updated_match_data], ignore_index=True)


X, y, df, label_encoder = prepare_data_for_training_binary(final)

df.to_csv('final2.csv')


