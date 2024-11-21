import pandas as pd
import requests
from io import StringIO
import os, time, random

# Rename any duplicated column names
def rename_duplicates(df):
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique(): 
        cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    return df

def determine_result(score):
    try:
        # Replace various types of dashes and special characters with a standard hyphen
        score = score.replace('–', '-').replace('—', '-').replace('−', '-')
        # Split the score into home and away scores
        home_score, away_score = map(int, score.split('-'))
        if home_score > away_score:
            return 'H'
        elif home_score < away_score:
            return 'A'
        else:
            return 'D'
    except Exception as e:
        print(f"Error processing score '{score}': {e}")
        return 'Invalid'

def fetch_and_process_data(season):
    # Determine the URLs based on the season
    if season == '2024/2025':
        url1 = "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"
        url2 = "https://fbref.com/en/comps/9/Premier-League-Stats"
    else:
        # Modify URL for past seasons
        season_formatted = season.replace('/', '-')
        url1 = f"https://fbref.com/en/comps/9/{season_formatted}/schedule/{season_formatted}-Premier-League-Scores-and-Fixtures"
        url2 = f"https://fbref.com/en/comps/9/{season_formatted}/{season_formatted}-Premier-League-Stats"

    # Fetch and process the scores and fixtures
    response = requests.get(url1)
    html_content = StringIO(response.text)
    tables = pd.read_html(html_content)

    if len(tables) >= 1:
        table1 = tables[0]
        table1 = table1.drop(columns=['Day', 'Date', 'Time', 'Attendance', 'Venue', 'Referee', 'Match Report', 'Notes'])
        table1 = table1.dropna()
        table1 = table1.dropna(axis=1)

        if 'Score' in table1.columns:
            table1['Result'] = table1['Score'].apply(determine_result)
            score_index = table1.columns.get_loc('Score')
            table1.insert(score_index + 1, 'Result', table1.pop('Result'))
        table1['Season'] = season
    else:
        print(f"There are more than one table on the Scores Page for season {season}.")
        return None, None

    # Fetch and process the stats
    response = requests.get(url2)
    html_content = StringIO(response.text)
    tables = pd.read_html(html_content)

    indices_to_keep = {0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22}
    filtered_tables = [table for i, table in enumerate(tables) if i in indices_to_keep]

    for i, table in enumerate(filtered_tables):
        if i != 0:
            table.columns = table.columns.droplevel(0)

    columns_to_delete_table_0 = ['Last 5', 'Attendance', 'Top Team Scorer', 'Goalkeeper', 'Notes']
    filtered_tables[0] = filtered_tables[0].drop(columns=[col for col in columns_to_delete_table_0 if col in filtered_tables[0].columns], errors='ignore')
    table_1_Columns = filtered_tables[1].columns
    new_table_1_Columns = ['Rk', 'Squad'] + [f'H{col}' for col in table_1_Columns[2:14]] + [f'A{col}' for col in table_1_Columns[14:]]
    filtered_tables[1].columns = new_table_1_Columns
    if filtered_tables[2].columns[22] == 'Gls':
        filtered_tables[2].drop(filtered_tables[2].columns[22], axis=1, inplace=True)

    combined_table = pd.DataFrame()

    for table in filtered_tables:
        columns_to_delete = ['90s', 'Starts', '# Pl', 'Min']
        table = table.drop(columns=[col for col in columns_to_delete if col in table.columns], errors='ignore')
        
        if 'Squad' not in table.columns:
            raise ValueError("'Squad' column is missing from one of the tables.")
        
        if combined_table.empty:
            combined_table = table
        else:
            columns_to_add = [col for col in table.columns if col not in combined_table.columns]
            if columns_to_add:
                combined_table = combined_table.merge(table[['Squad'] + columns_to_add], on='Squad', how='left')

    home_stats = combined_table.add_prefix(f'Home_')
    away_stats = combined_table.add_prefix(f'Away_')

    # Merge home and away stats with fixtures
    table1 = table1.merge(home_stats, how='left', left_on='Home', right_on='Home_Squad')
    table1 = table1.merge(away_stats, how='left', left_on='Away', right_on='Away_Squad')

    # Drop unnecessary columns (like 'Home_Squad' and 'Away_Squad') after merging
    table1 = table1.drop(columns=[f'Home_Squad', f'Away_Squad'])

    return table1

# List of seasons to process
seasons = ['2019/2020', '2020/2021', '2021/2022', '2022/2023', '2023/2024', '2024/2025']

all_tables = []

for season in seasons:
    # delay added to pass the bot detection from the website
    time.sleep(random.uniform(6, 10))
    table = fetch_and_process_data(season)
    if table is not None:
        all_tables.append(table)

# Concatenate all season data into one table
final_table = pd.concat(all_tables, ignore_index=True)
final_table = final_table.dropna(axis=1)
final_table = rename_duplicates(final_table)



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
final_table['Home_Form'] = 7.5  # Initial average form
final_table['Away_Form'] = 7.5  # Initial average form

# Iterate over each row (match) in the DataFrame
for idx, row in final_table.iterrows():
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
        final_table.at[idx, 'Home_Form'] = sum(team_form[home_team][-5:])
    if len(team_form[away_team]) > 5:
        final_table.at[idx, 'Away_Form'] = sum(team_form[away_team][-5:])



final_table['Home_Form2'] = 7.5  # Initial average form
final_table['Away_Form2'] = 7.5


# Initialize separate dictionaries for home and away forms
home_team_form = {}
away_team_form = {}

# Iterate over each row (match) in the DataFrame
for idx, row in final_table.iterrows():
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
        final_table.at[idx, 'Home_Form2'] = sum(home_team_form[home_team][-5:])
    if len(away_team_form[away_team]) > 5:
        final_table.at[idx, 'Away_Form2'] = sum(away_team_form[away_team][-5:])




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
final_table['Home_Goals_Last_3'] = 0
final_table['Away_Goals_Last_3'] = 0
final_table['Home_Goals_Conceded_Last_3'] = 0
final_table['Away_Goals_Conceded_Last_3'] = 0

# Iterate over each row in the dataframe (match data)
for idx, row in final_table.iterrows():
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
        final_table.at[idx, 'Home_Goals_Last_3'] = sum(team_goals[home_team]['scored'][-4:-1])  # sum last 3 excluding current game
    if len(team_goals[home_team]['conceded']) > 1:  # If there are at least 2 games, calculate
        final_table.at[idx, 'Home_Goals_Conceded_Last_3'] = sum(team_goals[home_team]['conceded'][-4:-1])  # sum last 3 excluding current game

    # For away team: calculate the total goals scored in the last 3 games excluding the current game
    if len(team_goals[away_team]['scored']) > 1:  # If there are at least 2 games, calculate
        final_table.at[idx, 'Away_Goals_Last_3'] = sum(team_goals[away_team]['scored'][-4:-1])  # sum last 3 excluding current game
    if len(team_goals[away_team]['conceded']) > 1:  # If there are at least 2 games, calculate
        final_table.at[idx, 'Away_Goals_Conceded_Last_3'] = sum(team_goals[away_team]['conceded'][-4:-1])  # sum last 3 excluding current game


# Optionally, save the modified data to a new CSV file
#Commented the below as it is not really helping


# # Initialize dictionaries to store head-to-head points for each matchup
# h2h_points = {}

# # Function to update the H2H points between two teams
# def update_h2h_points(home_team, away_team, home_points, away_points):
#     if (home_team, away_team) not in h2h_points:
#         h2h_points[(home_team, away_team)] = [0, 0]  # Initialize both teams with 0 points
#     h2h_points[(home_team, away_team)][0] += home_points
#     h2h_points[(home_team, away_team)][1] += away_points

# # Add columns for Home_H2H and Away_H2H
# final_table['Home_H2H'] = 0  # Initial H2H points for home team
# final_table['Away_H2H'] = 0  # Initial H2H points for away team

# # Iterate over each row (match) in the DataFrame
# for idx, row in final_table.iterrows():
#     home_team = row['Home']
#     away_team = row['Away']
#     result = row['Result']  # Full-time result (H, A, D)

#     # Initialize form for teams if they are not already in the dictionary
#     if (home_team, away_team) not in h2h_points:
#         h2h_points[(home_team, away_team)] = [0, 0]  # Initialize head-to-head points if teams have not faced

#     # Get current H2H points between these two teams
#     home_h2h = h2h_points[(home_team, away_team)][0]
#     away_h2h = h2h_points[(home_team, away_team)][1]

#     # Set the H2H values in the DataFrame
#     final_table.at[idx, 'Home_H2H'] = home_h2h
#     final_table.at[idx, 'Away_H2H'] = away_h2h

#     # Calculate points for the current match
#     home_points = get_points(result, is_home=True)
#     away_points = get_points(result, is_home=False)

#     # Update the head-to-head points between the home and away team
#     update_h2h_points(home_team, away_team, home_points, away_points)





# # Initialize dictionary to store head-to-head points for any matchup (regardless of home/away)
# h2h2_points = {}

# # Function to update H2H points between two teams, considering both home and away matchups
# def update_h2h2_points(team_a, team_b, points_a, points_b):
#     if (team_a, team_b) not in h2h2_points and (team_b, team_a) not in h2h2_points:
#         # Initialize the head-to-head record for both teams if they haven't faced each other
#         h2h2_points[(team_a, team_b)] = [0, 0]  # Points for team_a and team_b
#     # If the matchup already exists in either direction, sum points in both cases
#     if (team_a, team_b) in h2h2_points:
#         h2h2_points[(team_a, team_b)][0] += points_a
#         h2h2_points[(team_a, team_b)][1] += points_b
#     else:
#         h2h2_points[(team_b, team_a)][0] += points_b
#         h2h2_points[(team_b, team_a)][1] += points_a

# # Add columns for Home_H2H and Away_H2H
# final_table['Home_H2H2'] = 0  # Head-to-head points for the home team
# final_table['Away_H2H2'] = 0  # Head-to-head points for the away team

# # Iterate over each row (match) in the DataFrame
# for idx, row in final_table.iterrows():
#     home_team = row['Home']
#     away_team = row['Away']
#     result = row['Result']  # Full-time result (H, A, D)

#     # Determine points for the home and away team for the current match
#     home_points2 = get_points(result, is_home=True)
#     away_points2 = get_points(result, is_home=False)

#     # Get previous H2H points between these two teams (in any order)
#     if (home_team, away_team) in h2h2_points:
#         home_h2h2 = h2h2_points[(home_team, away_team)][0]
#         away_h2h2 = h2h2_points[(home_team, away_team)][1]
#     elif (away_team, home_team) in h2h2_points:
#         home_h2h2 = h2h2_points[(away_team, home_team)][1]
#         away_h2h2 = h2h2_points[(away_team, home_team)][0]
#     else:
#         home_h2h2 = 0
#         away_h2h2 = 0

#     # Set the H2H values in the DataFrame for the current row
#     final_table.at[idx, 'Home_H2H2'] = home_h2h2
#     final_table.at[idx, 'Away_H2H2'] = away_h2h2

#     # Update the head-to-head points between these two teams
#     update_h2h2_points(home_team, away_team, home_points2, away_points2)




final_table.to_csv('week12.csv', index=False)
print('done.')