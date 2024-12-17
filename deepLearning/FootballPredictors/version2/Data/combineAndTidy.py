import pandas as pd

def determine_result(df):
    return df.apply(
        lambda row: 'H' if row['home_team_goal_count'] > row['away_team_goal_count'] 
                    else 'A' if row['home_team_goal_count'] < row['away_team_goal_count'] 
                    else 'D',
        axis=1
    )

def drop_useless_columns_rename_matches(df):

    df['Result'] = determine_result(df)
    df = df.drop(columns=['timestamp', 'attendance', 'home_team_goal_timings', 'away_team_goal_timings', 'stadium_name'])
    df = df.rename(columns=
                   {"Pre-Match PPG (Home)": "home_team_PreMatch_PPG",
                    "Pre-Match PPG (Away)": "away_team_PreMatch_PPG",
                    "team_a_xg": "home_team_xg", "team_b_xg": "away_team_xg",
                    "home_ppg": "home_team_ppg", "away_ppg": "away_team_ppg",
                    "Home Team Pre-Match xG": "home_team_PreMatch_xG", "Away Team Pre-Match xG": "away_team_PreMatch_xG"
        })
    return df

def drop_useless_columns_rename_teamStats(df):
    df = df.drop(columns=['team_name', 'country'])
    return df

seasons = ['2020/2021', '2021/2022', '2022/2023', '2023/2024', '2024/2025']
combinedData = []

for season in seasons:

    matches = pd.read_csv(f'{season.split('/')[0]}/england-premier-league-matches-{season.replace("/", "-to-")}-stats.csv')
    teamStats = pd.read_csv(f'{season.split('/')[0]}/england-premier-league-teams-{season.replace("/", "-to-")}-stats.csv')

    matches = drop_useless_columns_rename_matches(matches)
    teamStats = drop_useless_columns_rename_teamStats(teamStats)

    homeStats = teamStats.add_prefix('home_team_')
    awayStats = teamStats.add_prefix('away_team_')

    matches = matches.merge(homeStats, how='left', left_on='home_team_name', right_on='home_team_common_name')
    matches = matches.merge(awayStats, how='left', left_on='away_team_name', right_on='away_team_common_name')

    matches = matches.drop(columns=["home_team_common_name", "away_team_common_name"])

    combinedData.append(matches)

final = pd.concat(combinedData, ignore_index=True)
final.to_csv('final.csv', index=False)

