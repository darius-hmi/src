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

def apply_label_encoder(df):
    label_encoder = LabelEncoder()
    all_teams = pd.concat([df['Home'], df['Away']]).unique()
    label_encoder.fit(all_teams)
    df['Home'] = label_encoder.transform(df['Home'])
    df['Away'] = label_encoder.transform(df['Away'])
    return df, label_encoder

def apply_one_hot_encoder(df):
    one_hot_encoder_teams = OneHotEncoder(sparse_output=False)
    # One-hot encode the Home teams
    home_teams = df['Home'].unique().reshape(-1, 1)  # Get unique home teams
    home_encoded = one_hot_encoder_teams.fit_transform(home_teams)
    home_encoded_df = pd.DataFrame(home_encoded, columns=one_hot_encoder_teams.get_feature_names_out(['Home_Team']))
    # One-hot encode the Away teams
    away_teams = df['Away'].unique().reshape(-1, 1)  # Get unique away teams
    away_encoded = one_hot_encoder_teams.fit_transform(away_teams)
    away_encoded_df = pd.DataFrame(away_encoded, columns=one_hot_encoder_teams.get_feature_names_out(['Away_Team']))

    home_mapping = {team: home_encoded[i] for i, team in enumerate(df['Home'].unique())}
    away_mapping = {team: away_encoded[i] for i, team in enumerate(df['Away'].unique())}

    df['Home_encoded'] = df['Home'].apply(lambda x: home_mapping[x])
    df['Away_encoded'] = df['Away'].apply(lambda x: away_mapping[x])
    df = pd.concat([df, pd.DataFrame(df['Home_encoded'].tolist(), columns=home_encoded_df.columns)], axis=1)
    df = pd.concat([df, pd.DataFrame(df['Away_encoded'].tolist(), columns=away_encoded_df.columns)], axis=1)
    df.drop(columns=['Home', 'Away', 'Home_encoded', 'Away_encoded'], inplace=True)
    return df