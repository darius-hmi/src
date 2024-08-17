import pandas as pd
import os

match_results = pd.read_csv('cleaned_match_results.csv')
standings = pd.read_csv('cleaned_table_standings.csv')
# player_stats = pd.read_csv('cleaned_player_stats.csv')
# player_wages = pd.read_csv('cleaned_player_wages.csv')

# team_stats = player_stats.groupby(['Team']).agg({
#     'totalShots': 'sum',
#     'yellowCards': 'sum',
#     'redCards': 'sum',
#     'foulsCommitted': 'sum',
#     'foulsSuffered': 'sum',
#     'Total Play Time(min)': 'mean',
#     'ownGoals': 'sum',
#     'offsides': 'sum',
#     'goalAssists': 'sum',
#     'shotsOnTarget': 'sum',
# }).reset_index()

# Rename columns in standings to differentiate home and away
standings_home = standings.rename(columns={col: f"home_{col}" for col in standings.columns if col not in ['Team', 'season']})
standings_away = standings.rename(columns={col: f"away_{col}" for col in standings.columns if col not in ['Team', 'season']})

# Merge home and away standings with match results
match_results = pd.merge(match_results, standings_home, left_on=['HomeTeam', 'season'], right_on=['Team', 'season'], how='left')
match_results = pd.merge(match_results, standings_away, left_on=['AwayTeam', 'season'], right_on=['Team', 'season'], how='left')

print(match_results)

match_results.to_csv('temp.csv', index=False)





