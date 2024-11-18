import pandas as pd

# Load your CSV file into a DataFrame
df = pd.read_csv('final45.csv')

# Initialize a dictionary to store the rolling points for each team
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
df['Home_Form'] = 7.5  # Initial average form
df['Away_Form'] = 7.5  # Initial average form

# Iterate over each row (match) in the DataFrame
for idx, row in df.iterrows():
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
        df.at[idx, 'Home_Form'] = sum(team_form[home_team][-5:])
    if len(team_form[away_team]) > 5:
        df.at[idx, 'Away_Form'] = sum(team_form[away_team][-5:])



df['Home_Form2'] = 7.5  # Initial average form
df['Away_Form2'] = 7.5


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
        df.at[idx, 'Home_Form2'] = sum(home_team_form[home_team][-5:])
    if len(away_team_form[away_team]) > 5:
        df.at[idx, 'Away_Form2'] = sum(away_team_form[away_team][-5:])




# Initialize dictionaries to store head-to-head points for each matchup
h2h_points = {}

# Function to update the H2H points between two teams
def update_h2h_points(home_team, away_team, home_points, away_points):
    if (home_team, away_team) not in h2h_points:
        h2h_points[(home_team, away_team)] = [0, 0]  # Initialize both teams with 0 points
    h2h_points[(home_team, away_team)][0] += home_points
    h2h_points[(home_team, away_team)][1] += away_points

# Add columns for Home_H2H and Away_H2H
df['Home_H2H'] = 0  # Initial H2H points for home team
df['Away_H2H'] = 0  # Initial H2H points for away team

# Iterate over each row (match) in the DataFrame
for idx, row in df.iterrows():
    home_team = row['Home']
    away_team = row['Away']
    result = row['Result']  # Full-time result (H, A, D)

    # Initialize form for teams if they are not already in the dictionary
    if (home_team, away_team) not in h2h_points:
        h2h_points[(home_team, away_team)] = [0, 0]  # Initialize head-to-head points if teams have not faced

    # Get current H2H points between these two teams
    home_h2h = h2h_points[(home_team, away_team)][0]
    away_h2h = h2h_points[(home_team, away_team)][1]

    # Set the H2H values in the DataFrame
    df.at[idx, 'Home_H2H'] = home_h2h
    df.at[idx, 'Away_H2H'] = away_h2h

    # Calculate points for the current match
    home_points = get_points(result, is_home=True)
    away_points = get_points(result, is_home=False)

    # Update the head-to-head points between the home and away team
    update_h2h_points(home_team, away_team, home_points, away_points)





# Initialize dictionary to store head-to-head points for any matchup (regardless of home/away)
h2h2_points = {}

# Function to update H2H points between two teams, considering both home and away matchups
def update_h2h2_points(team_a, team_b, points_a, points_b):
    if (team_a, team_b) not in h2h2_points and (team_b, team_a) not in h2h2_points:
        # Initialize the head-to-head record for both teams if they haven't faced each other
        h2h2_points[(team_a, team_b)] = [0, 0]  # Points for team_a and team_b
    # If the matchup already exists in either direction, sum points in both cases
    if (team_a, team_b) in h2h2_points:
        h2h2_points[(team_a, team_b)][0] += points_a
        h2h2_points[(team_a, team_b)][1] += points_b
    else:
        h2h2_points[(team_b, team_a)][0] += points_b
        h2h2_points[(team_b, team_a)][1] += points_a

# Add columns for Home_H2H and Away_H2H
df['Home_H2H2'] = 0  # Head-to-head points for the home team
df['Away_H2H2'] = 0  # Head-to-head points for the away team

# Iterate over each row (match) in the DataFrame
for idx, row in df.iterrows():
    home_team = row['Home']
    away_team = row['Away']
    result = row['Result']  # Full-time result (H, A, D)

    # Determine points for the home and away team for the current match
    home_points2 = get_points(result, is_home=True)
    away_points2 = get_points(result, is_home=False)

    # Get previous H2H points between these two teams (in any order)
    if (home_team, away_team) in h2h2_points:
        home_h2h2 = h2h2_points[(home_team, away_team)][0]
        away_h2h2 = h2h2_points[(home_team, away_team)][1]
    elif (away_team, home_team) in h2h2_points:
        home_h2h2 = h2h2_points[(away_team, home_team)][1]
        away_h2h2 = h2h2_points[(away_team, home_team)][0]
    else:
        home_h2h2 = 0
        away_h2h2 = 0

    # Set the H2H values in the DataFrame for the current row
    df.at[idx, 'Home_H2H2'] = home_h2h2
    df.at[idx, 'Away_H2H2'] = away_h2h2

    # Update the head-to-head points between these two teams
    update_h2h2_points(home_team, away_team, home_points2, away_points2)


# Save the updated DataFrame to a new CSV
df.to_csv('final45_newFeatures.csv', index=False)
