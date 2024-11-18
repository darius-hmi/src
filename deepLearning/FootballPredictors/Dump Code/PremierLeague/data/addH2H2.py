import pandas as pd

# Load your CSV file into a DataFrame
df = pd.read_csv('final_New.csv')

# Initialize a dictionary to store the rolling points for each team
def get_points(result, is_home):
    if result == 'H':  # Home win
        return 3 if is_home else 0
    elif result == 'A':  # Away win
        return 0 if is_home else 3
    elif result == 'D':  # Draw
        return 1
    else:
        return 0  # In case of missing or incorrect result



# Initialize dictionary to store head-to-head points for any matchup (regardless of home/away)
h2h_points = {}

# Function to update H2H points between two teams, considering both home and away matchups
def update_h2h_points(team_a, team_b, points_a, points_b):
    if (team_a, team_b) not in h2h_points and (team_b, team_a) not in h2h_points:
        # Initialize the head-to-head record for both teams if they haven't faced each other
        h2h_points[(team_a, team_b)] = [0, 0]  # Points for team_a and team_b
    # If the matchup already exists in either direction, sum points in both cases
    if (team_a, team_b) in h2h_points:
        h2h_points[(team_a, team_b)][0] += points_a
        h2h_points[(team_a, team_b)][1] += points_b
    else:
        h2h_points[(team_b, team_a)][0] += points_b
        h2h_points[(team_b, team_a)][1] += points_a

# Add columns for Home_H2H and Away_H2H
df['Home_H2H'] = 0  # Head-to-head points for the home team
df['Away_H2H'] = 0  # Head-to-head points for the away team

# Iterate over each row (match) in the DataFrame
for idx, row in df.iterrows():
    home_team = row['Home']
    away_team = row['Away']
    result = row['Result']  # Full-time result (H, A, D)

    # Determine points for the home and away team for the current match
    home_points = get_points(result, is_home=True)
    away_points = get_points(result, is_home=False)

    # Get previous H2H points between these two teams (in any order)
    if (home_team, away_team) in h2h_points:
        home_h2h = h2h_points[(home_team, away_team)][0]
        away_h2h = h2h_points[(home_team, away_team)][1]
    elif (away_team, home_team) in h2h_points:
        home_h2h = h2h_points[(away_team, home_team)][1]
        away_h2h = h2h_points[(away_team, home_team)][0]
    else:
        home_h2h = 0
        away_h2h = 0

    # Set the H2H values in the DataFrame for the current row
    df.at[idx, 'Home_H2H'] = home_h2h
    df.at[idx, 'Away_H2H'] = away_h2h

    # Update the head-to-head points between these two teams
    update_h2h_points(home_team, away_team, home_points, away_points)

# Save the updated DataFrame to a new CSV
df.to_csv('matches_with_h2h2.csv', index=False)
