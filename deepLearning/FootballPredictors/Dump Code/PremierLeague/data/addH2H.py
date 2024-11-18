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

# Save the updated DataFrame to a new CSV
df.to_csv('matches_with_h2h.csv', index=False)
