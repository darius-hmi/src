import pandas as pd


final_table = pd.read_csv('finalMini.csv')


team_form = {}

def get_points(result, is_home):
    if result == 'H':  # Home win
        return 3 if is_home else 0
    elif result == 'A':  # Away win
        return 0 if is_home else 3
    elif result == 'D':  # Draw
        return 1
    else:
        return 0 

final_table['home_team_Form'] = 7.5  
final_table['away_team_Form'] = 7.5  

for idx, row in final_table.iterrows():
    home_team = row['home_team_name']
    away_team = row['away_team_name']
    result = row['Result'] 

    if home_team not in team_form:
        team_form[home_team] = []
    if away_team not in team_form:
        team_form[away_team] = []

    home_points = get_points(result, is_home=True)
    away_points = get_points(result, is_home=False)

    team_form[home_team].append(home_points)
    team_form[away_team].append(away_points)

    if len(team_form[home_team]) > 5:
        final_table.at[idx, 'home_team_Form'] = sum(team_form[home_team][-6:-1])
    if len(team_form[away_team]) > 5:
        final_table.at[idx, 'away_team_Form'] = sum(team_form[away_team][-6:-1])



final_table['home_team_Form2'] = 7.5 
final_table['away_team_Form2'] = 7.5


home_team_form = {}
away_team_form = {}

for i, row in final_table.iterrows():
    home_team = row['home_team_name']
    away_team = row['away_team_name']
    result = row['Result'] 

    if home_team not in home_team_form:
        home_team_form[home_team] = []
    if away_team not in away_team_form:
        away_team_form[away_team] = []

    home_points = get_points(result, is_home=True)
    away_points = get_points(result, is_home=False)

    home_team_form[home_team].append(home_points)
    away_team_form[away_team].append(away_points)

    if len(home_team_form[home_team]) > 5:
        final_table.at[i, 'home_team_Form2'] = sum(home_team_form[home_team][-6:-1])
    if len(away_team_form[away_team]) > 5:
        final_table.at[i, 'away_team_Form2'] = sum(away_team_form[away_team][-6:-1])

team_goals = {}

final_table['home_team_Goals_Last_3'] = 0
final_table['away_team_Goals_Last_3'] = 0
final_table['home_team_Goals_Conceded_Last_3'] = 0
final_table['away_team_Goals_Conceded_Last_3'] = 0

for j, row in final_table.iterrows():
    home_team = row['home_team_name']
    away_team = row['away_team_name']
    home_goals = row['home_team_goal_count']
    away_goals = row['away_team_goal_count']


    if home_team not in team_goals:
        team_goals[home_team] = {'scored': [], 'conceded': []}
    if away_team not in team_goals:
        team_goals[away_team] = {'scored': [], 'conceded': []}

    team_goals[home_team]['scored'].append(home_goals)
    team_goals[away_team]['scored'].append(away_goals)
    team_goals[home_team]['conceded'].append(away_goals)
    team_goals[away_team]['conceded'].append(home_goals)

    if len(team_goals[home_team]['scored']) > 3: 
        final_table.at[j, 'home_team_Goals_Last_3'] = sum(team_goals[home_team]['scored'][-4:-1])  
    if len(team_goals[home_team]['conceded']) > 3:
        final_table.at[j, 'home_team_Goals_Conceded_Last_3'] = sum(team_goals[home_team]['conceded'][-4:-1])  

    if len(team_goals[away_team]['scored']) > 3: 
        final_table.at[j, 'away_team_Goals_Last_3'] = sum(team_goals[away_team]['scored'][-4:-1])  
    if len(team_goals[away_team]['conceded']) > 3:
        final_table.at[j, 'away_team_Goals_Conceded_Last_3'] = sum(team_goals[away_team]['conceded'][-4:-1])


final_table.to_csv('finalMiniPlus.csv', index=False)