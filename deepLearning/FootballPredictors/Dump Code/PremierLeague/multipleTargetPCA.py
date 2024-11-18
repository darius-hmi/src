import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping

def extract_start_year(season_str):
    return int(season_str.split('/')[0])

def exponential_decay_weight(df, decay_factor):
    df['Season_Year'] = df['Season'].apply(extract_start_year)
    max_season = df['Season_Year'].max()
    weights = np.exp(-decay_factor * (max_season - df['Season_Year']))
    return weights

def prepare_data(file_path, decay_factor=0.1, n_components=50):
    match_result = pd.read_csv(file_path)
    match_result['Weight'] = exponential_decay_weight(match_result, decay_factor)
    match_result = match_result.drop(columns=['Season_Year'])

    label_encoder_home = LabelEncoder()
    label_encoder_away = LabelEncoder()
    match_result['Home'] = label_encoder_home.fit_transform(match_result['Home'])
    match_result['Away'] = label_encoder_away.fit_transform(match_result['Away'])

    home_prefix = 'Home_'
    away_prefix = 'Away_'
    additional_home_columns = ['xG']
    additional_away_columns = ['xG.1']

    home_columns = [col for col in match_result.columns if col.startswith(home_prefix)] + additional_home_columns
    away_columns = [col for col in match_result.columns if col.startswith(away_prefix)] + additional_away_columns

    def calculate_weighted_averages(df, group_by_col, columns, weight_col='Weight'):
        return df.groupby(group_by_col).apply(
            lambda x: pd.Series({
                col: np.average(x[col], weights=x[weight_col]) if col in x.columns else np.nan
                for col in columns
            })
        ).reset_index()

    home_stats = calculate_weighted_averages(match_result, 'Home', home_columns)
    away_stats = calculate_weighted_averages(match_result, 'Away', away_columns)

    target = match_result['Away_(\'Performance\', \'SoT\')']  # Adjust to your actual target column name
    features = match_result[['Wk', 'Home', 'Away'] + home_columns + away_columns]

    scalar = StandardScaler()
    features_scaled = scalar.fit_transform(features)

    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features_scaled)

    return features_pca, target, label_encoder_home, label_encoder_away, scalar, pca, home_stats, away_stats

def build_and_train_model(X_train, Y_train):
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=regularizers.l2(0.2)))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.2)))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='linear'))  # Regression output
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

    return model

def cross_validate_model(features, target, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    mae_scores = []

    for train_index, test_index in kf.split(features):
        X_train, X_test = features[train_index], features[test_index]
        Y_train, Y_test = target[train_index], target[test_index]

        model = build_and_train_model(X_train, Y_train)
        predictions = model.predict(X_test)
        mae = np.mean(np.abs(Y_test - predictions.flatten()))  # Calculate Mean Absolute Error
        mae_scores.append(mae)

    return np.mean(mae_scores)

def predict_match_result(model, scalar, pca, label_encoder_home, label_encoder_away, home_stats, away_stats, home_team, away_team, wk):
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

    example_match_scaled = scalar.transform(example_match)
    example_match_pca = pca.transform(example_match_scaled)

    predicted_shots_on_target = model.predict(example_match_pca)[0][0]

    return predicted_shots_on_target

if __name__ == "__main__":
    features_pca, target, label_encoder_home, label_encoder_away, scalar, pca, home_stats, away_stats = prepare_data('data/final_lessDecisiveStats.csv', decay_factor=0.1, n_components=50)

    avg_mae = cross_validate_model(features_pca, target, k=5)
    print(f'Average Mean Absolute Error (MAE) from cross-validation: {avg_mae:.2f}')

    X_train, X_test, y_train, y_test = train_test_split(features_pca, target, test_size=0.2, random_state=42)
    model = build_and_train_model(X_train, y_train)

    home_teams = ['Fulham', 'Ipswich Town', 'Manchester Utd', 'Newcastle Utd', 'Southampton', 'Tottenham']
    away_teams = ['Aston Villa', 'Everton', 'Brentford', 'Brighton', 'Leicester City', 'West Ham'] 
    wk = '8' 

    for home_team, away_team in zip(home_teams, away_teams):
        predicted_shots_on_target = predict_match_result(model, scalar, pca, label_encoder_home, label_encoder_away, home_stats, away_stats, home_team, away_team, wk)
        print(f"Predicted shots on target for {home_team} vs. {away_team}: {predicted_shots_on_target:.2f}")
