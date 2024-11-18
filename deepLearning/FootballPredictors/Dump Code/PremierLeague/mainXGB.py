import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np

def extract_start_year(season_str):
    """Extract the starting year from the season string in 'YYYY/YYYY' format."""
    return int(season_str.split('/')[0])

def exponential_decay_weight(df, decay_factor):
    """Apply exponential decay to the season weights."""    
    df['Season_Year'] = df['Season'].apply(extract_start_year)
    max_season = df['Season_Year'].max()  # Determine the most recent season
    weights = np.exp(-decay_factor * (max_season - df['Season_Year']))
    return weights

def prepare_data(file_path, decay_factor=0.1):
    """Load and prepare the dataset with time decay."""
    match_results = pd.read_csv(file_path)
    # Apply exponential decay to the weights
    match_results['Weight'] = exponential_decay_weight(match_results, decay_factor)
    match_results = match_results.drop(columns=['Season_Year'])

    # Map FTR values to numeric (Home win: 1, Draw: 0, Away win: 2)
    ftr_map = {'H': 1, 'D': 0, 'A': 2}
    match_results['Result'] = match_results['Result'].map(ftr_map)

    # Encode team names
    label_encoder_home = LabelEncoder()
    label_encoder_away = LabelEncoder()
    match_results['Home'] = label_encoder_home.fit_transform(match_results['Home'])
    match_results['Away'] = label_encoder_away.fit_transform(match_results['Away'])

    home_prefix = 'Home_'
    away_prefix = 'Away_'
    additional_home_columns = ['xG']
    additional_away_columns = ['xG.1']

    home_columns = [col for col in match_results.columns if col.startswith(home_prefix)] + additional_home_columns
    away_columns = [col for col in match_results.columns if col.startswith(away_prefix)] + additional_away_columns

    def calculate_weighted_averages(df, group_by_col, columns, weight_col='Weight'):
        return df.groupby(group_by_col).apply(
            lambda x: pd.Series({
                col: np.average(x[col], weights=x[weight_col]) if col in x.columns else np.nan
                for col in columns
            })
        ).reset_index()

    # Calculate weighted averages for home stats including xG
    home_stats = calculate_weighted_averages(match_results, 'Home', home_columns)

    # Calculate weighted averages for away stats including xG.1
    away_stats = calculate_weighted_averages(match_results, 'Away', away_columns)

    features = match_results[[ 'Wk', 'Home', 'Away'] + home_columns + away_columns ]
    target = match_results['Result']

    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled, target, label_encoder_home, label_encoder_away, scaler, home_stats, away_stats

def build_and_train_model(X_train, y_train, alpha=0.1, lambda_=2.0):
    """Build and train the XGBoost model with L1 and L2 regularization."""
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        learning_rate=0.013,  # Default learning rate
        max_depth=10,        # A common value for depth
        min_child_weight=1,  # Default value
        subsample=0.8,       # Default subsampling rate
        colsample_bytree=0.8,  # Default colsample_bytree
        use_label_encoder=False,
        alpha=alpha,  # L1 regularization term
        lambda_=lambda_  # L2 regularization term
    )
    model.fit(X_train, y_train)
    return model

def predict_match_result(model, scaler, label_encoder_home, label_encoder_away, home_stats, away_stats, home_team, away_team):
    """Predict the match result for given teams and return probabilities of all outcomes."""
    # Encode team names
    home_team_encoded = label_encoder_home.transform([home_team])[0]
    away_team_encoded = label_encoder_away.transform([away_team])[0]

    home_stats_row = home_stats.loc[home_team_encoded]
    away_stats_row = away_stats.loc[away_team_encoded]

    example_match = pd.DataFrame({
        'Wk': [wk],
        'Home': [home_team_encoded],
        'Away': [away_team_encoded],
        **home_stats_row.to_dict(),
        **away_stats_row.to_dict()
    })

    # Normalize example match data
    example_match_scaled = scaler.transform(example_match)

    # Predict probabilities
    predictions = model.predict_proba(example_match_scaled)

    # Extract probabilities for Home Win, Draw, and Away Win
    draw_probability = predictions[0][0]      # Probability of Draw
    home_win_probability = predictions[0][1]  # Probability of Home Win
    away_win_probability = predictions[0][2]  # Probability of Away Win

    return home_win_probability, draw_probability, away_win_probability

# Main workflow
if __name__ == "__main__":
    # Load and prepare data with decay factor
    features_scaled, target, label_encoder_home, label_encoder_away, scaler, home_stats, away_stats = prepare_data('data/final45_newFeatures.csv', decay_factor=0.1)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

    # Build and train the model using the build_and_train_model function
    xgb_model = build_and_train_model(X_train, y_train, alpha=0.1, lambda_=2.0)

    home_teams = ['Brentford', 'Crystal Palace', 'West Ham', 'Wolves', 'Brighton', 'Liverpool', 'Manchester Utd', 'Nott\'ham Forest', 'Tottenham', 'Chelsea']
    away_teams = ['Bournemouth', 'Fulham', 'Everton', 'Southampton', 'Manchester City', 'Aston Villa', 'Leicester City', 'Newcastle Utd', 'Ipswich Town', 'Arsenal'] 
    wk = '11' 

    for home_team, away_team in zip(home_teams, away_teams):
        home_win_probability, draw_probability, away_win_probability = predict_match_result(
            xgb_model, scaler, label_encoder_home, label_encoder_away, home_stats, away_stats, home_team, away_team
        )

        print(f"Win probabilities for {home_team} vs. {away_team}:")
        print(f"Home Win: {home_win_probability:.2f}")
        print(f"Draw: {draw_probability:.2f}")
        print(f"Away Win: {away_win_probability:.2f}")

