import pandas as pd

# Load your CSV file into a DataFrame
df = pd.read_csv('final_New.csv')

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


# Initialize separate dictionaries for home and away forms
home_team_form = {}
away_team_form = {}

# Iterate over each row (match) in the DataFrame
for idx, row in df.iterrows():
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
        df.at[idx, 'HomeForm'] = sum(home_team_form[home_team][-5:])
    if len(away_team_form[away_team]) > 5:
        df.at[idx, 'AwayForm'] = sum(away_team_form[away_team][-5:])

df.to_csv('matches_with_form2.csv', index=False)

