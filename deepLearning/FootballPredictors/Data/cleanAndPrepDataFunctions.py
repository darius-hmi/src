import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def order_features_and_prepare_target(df):
    df = df.rename(columns={'xG': 'Home_xG_Base', 'xG.1': 'Away_xG_Base'})
    home_prefix = 'Home_'
    away_prefix = 'Away_'
    home_columns = [col for col in df.columns if col.startswith(home_prefix)]
    away_columns = [col for col in df.columns if col.startswith(away_prefix)]
    X = df[[
        'Wk', 'Home', 'Away'] + home_columns + away_columns
    ]
    y = df['Result']
    df = pd.concat([X, y], axis=1)

    return X, y, df

def order_features_and_prepare_target_oneHot(df):
    # X = df.drop(columns=['Result'])
    # y = df['Result']
    # df = pd.concat([X, y], axis=1)

    # return X, y, df

    df = df.rename(columns={'xG': 'Home_xG_Base', 'xG.1': 'Away_xG_Base'})
    home_prefix = 'Home_'
    away_prefix = 'Away_'
    home_columns = [col for col in df.columns if col.startswith(home_prefix)]
    away_columns = [col for col in df.columns if col.startswith(away_prefix)]
    X = df[[
        'Wk'] + home_columns + away_columns
    ]
    y = df['Result']
    df = pd.concat([X, y], axis=1)

    return X, y, df


def drop_seaon_col(df):
    df = df.drop(columns=['Season'])
    return df

def apply_scoreToResult_012(df):
    ftr_map = {'H': 1, 'D': 0, 'A': 2}
    df['Result'] = df['Result'].map(ftr_map)
    return df

def apply_scoreToResult_01minus1(df):
    ftr_map = {'H': 1, 'D': 0, 'A': -1}
    df['Result'] = df['Result'].map(ftr_map)
    return df

def apply_scoreToResult_binary(df):
    ftr_map = {'H': 1, 'D': 0, 'A': 0}
    df['Result'] = df['Result'].map(ftr_map)
    return df

def apply_label_encoder(df):
    label_encoder = LabelEncoder()
    all_teams = pd.concat([df['Home'], df['Away']]).unique()
    label_encoder.fit(all_teams)
    df['Home'] = label_encoder.transform(df['Home'])
    df['Away'] = label_encoder.transform(df['Away'])
    return df, label_encoder


def apply_one_hot_encoder(df):
    # Combine "Home" and "Away" columns for consistent encoding
    teams = pd.concat([df['Home'], df['Away']]).unique().reshape(-1, 1)
    
    # One-hot encode all teams
    label_encoder = OneHotEncoder(sparse_output=False)
    encoded_teams = label_encoder.fit_transform(teams)
    
    # Create a mapping for team names
    team_mapping = {team: encoded_teams[i] for i, team in enumerate(teams.ravel())}
    
    # Apply one-hot encoding to "Home" and "Away" columns
    home_encoded = df['Home'].apply(lambda x: team_mapping[x]).tolist()
    away_encoded = df['Away'].apply(lambda x: team_mapping[x]).tolist()
    
    # Create DataFrame for the encoded features
    home_encoded_df = pd.DataFrame(home_encoded, columns=label_encoder.get_feature_names_out(['Team']))
    away_encoded_df = pd.DataFrame(away_encoded, columns=label_encoder.get_feature_names_out(['Team']))
    
    # Add "Home_" and "Away_" prefixes to distinguish columns
    home_encoded_df = home_encoded_df.add_prefix('Home_')
    away_encoded_df = away_encoded_df.add_prefix('Away_')
    
    # Concatenate encoded features with original data
    df = pd.concat([df, home_encoded_df, away_encoded_df], axis=1)
    
    # Drop the original "Home" and "Away" columns
    df.drop(columns=['Home', 'Away'], inplace=True)
    
    return df, label_encoder


def prepare_data_for_training(df):
    df, label_encoder = apply_label_encoder(df)
    df = drop_seaon_col(df)
    df = apply_scoreToResult_012(df)
    X, y, df = order_features_and_prepare_target(df)

    return X, y, df, label_encoder

def prepare_data_for_training_binary(df):
    df, label_encoder = apply_label_encoder(df)
    df = drop_seaon_col(df)
    df = apply_scoreToResult_binary(df)
    X, y, df = order_features_and_prepare_target(df)

    return X, y, df, label_encoder

def prepare_data_for_training_oneHot(df):
    df, label_encoder = apply_one_hot_encoder(df)
    df = drop_seaon_col(df)
    df = apply_scoreToResult_binary(df)
    X, y, df = order_features_and_prepare_target_oneHot(df)

    return X, y, df, label_encoder


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
        
def prepare_match_data(df, home_team, away_team, label_encoder):
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


def prepare_match_data_hack(df, home_team, away_team):


    wk = df['Wk'].iloc[-1] + 1

    match_data_hack = pd.DataFrame({
        'Wk': [wk],
        'Home': [home_team],
        'Away': [away_team],
    })
    return match_data_hack



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


def get_goals_from_score(score):
    try:
        score = score.replace('–', '-').replace('—', '-').replace('−', '-')
        home_score, away_score = map(int, score.split('-'))
        return home_score, away_score
    except Exception as e:
        print(f"Error processing score '{score}': {e}")
        return None, None


def apply_form_and_last3_goals(df):
    team_form = {}
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
    

    # Initialize a dictionary to keep track of the last 3 games' goals for each team
    team_goals = {}

    # Example of how you might load your data
    # Add new columns to store the total goals scored and conceded in the last 3 games
    df['Home_Goals_Last_3'] = 0
    df['Away_Goals_Last_3'] = 0
    df['Home_Goals_Conceded_Last_3'] = 0
    df['Away_Goals_Conceded_Last_3'] = 0

    # Iterate over each row in the dataframe (match data)
    for idx, row in df.iloc[:-1].iterrows():
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
            df.at[idx, 'Home_Goals_Last_3'] = sum(team_goals[home_team]['scored'][-4:-1])  # sum last 3 excluding current game
        if len(team_goals[home_team]['conceded']) > 1:  # If there are at least 2 games, calculate
            df.at[idx, 'Home_Goals_Conceded_Last_3'] = sum(team_goals[home_team]['conceded'][-4:-1])  # sum last 3 excluding current game

        # For away team: calculate the total goals scored in the last 3 games excluding the current game
        if len(team_goals[away_team]['scored']) > 1:  # If there are at least 2 games, calculate
            df.at[idx, 'Away_Goals_Last_3'] = sum(team_goals[away_team]['scored'][-4:-1])  # sum last 3 excluding current game
        if len(team_goals[away_team]['conceded']) > 1:  # If there are at least 2 games, calculate
            df.at[idx, 'Away_Goals_Conceded_Last_3'] = sum(team_goals[away_team]['conceded'][-4:-1])  # sum last 3 excluding current game

    last_row_idx = df.index[-1]
    last_row = df.iloc[-1]
    home_team = last_row['Home']
    away_team = last_row['Away']

    # Compute values based on the last 3 matches
    if len(team_goals[home_team]['scored']) >= 3:
        df.at[last_row_idx, 'Home_Goals_Last_3'] = sum(team_goals[home_team]['scored'][-3:])
    if len(team_goals[home_team]['conceded']) >= 3:
        df.at[last_row_idx, 'Home_Goals_Conceded_Last_3'] = sum(team_goals[home_team]['conceded'][-3:])

    if len(team_goals[away_team]['scored']) >= 3:
        df.at[last_row_idx, 'Away_Goals_Last_3'] = sum(team_goals[away_team]['scored'][-3:])
    if len(team_goals[away_team]['conceded']) >= 3:
        df.at[last_row_idx, 'Away_Goals_Conceded_Last_3'] = sum(team_goals[away_team]['conceded'][-3:])
    
    return df