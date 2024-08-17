import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

def exponential_decay_weight(df, decay_factor):
    """Apply exponential decay to the season weights."""
    max_season = df['Season'].max()
    weights = np.exp(-decay_factor * (max_season - df['Season']))
    return weights

def prepare_data(file_path, decay_factor=0.1):
    """Load and prepare the dataset with time decay."""
    match_results = pd.read_csv(file_path)
    match_results['Season'] = pd.to_datetime(match_results['Date']).dt.year

    # Apply exponential decay to the weights
    match_results['Weight'] = exponential_decay_weight(match_results, decay_factor)

    # Map FTR values to numeric (Home win: 1, Draw: 0, Away win: 2)
    ftr_map = {'H': 1, 'D': 0, 'A': 2}
    match_results['FTR'] = match_results['FTR'].map(ftr_map)

    # Calculate weighted average statistics for home and away teams
    home_stats = match_results.groupby('HomeTeam').apply(
        lambda x: pd.Series({
            'home_Pts': np.average(x['home_Pts'], weights=x['Weight']),
            'home_HP': np.average(x['home_HP'], weights=x['Weight']),
            'home_HW': np.average(x['home_HW'], weights=x['Weight']),
            'home_HD': np.average(x['home_HD'], weights=x['Weight']),
            'home_HL': np.average(x['home_HL'], weights=x['Weight']),
            'home_AP': np.average(x['home_AP'], weights=x['Weight']),
            'home_AW': np.average(x['home_AW'], weights=x['Weight']),
            'home_AD': np.average(x['home_AD'], weights=x['Weight']),
            'home_AL': np.average(x['home_AL'], weights=x['Weight']),
            'home_TP': np.average(x['home_TP'], weights=x['Weight']),
            'home_TW': np.average(x['home_TW'], weights=x['Weight']),
            'home_TD': np.average(x['home_TD'], weights=x['Weight']),
            'home_TL': np.average(x['home_TL'], weights=x['Weight']),
            'home_TF': np.average(x['home_TF'], weights=x['Weight']),
            'home_TA': np.average(x['home_TA'], weights=x['Weight']),
            'home_TAG': np.average(x['home_TAG'], weights=x['Weight']),
        })
    ).reset_index()
    
    away_stats = match_results.groupby('AwayTeam').apply(
        lambda x: pd.Series({
            'away_Pts': np.average(x['away_Pts'], weights=x['Weight']),
            'away_HP': np.average(x['away_HP'], weights=x['Weight']),
            'away_HW': np.average(x['away_HW'], weights=x['Weight']),
            'away_HD': np.average(x['away_HD'], weights=x['Weight']),
            'away_HL': np.average(x['away_HL'], weights=x['Weight']),
            'away_AP': np.average(x['away_AP'], weights=x['Weight']),
            'away_AW': np.average(x['away_AW'], weights=x['Weight']),
            'away_AD': np.average(x['away_AD'], weights=x['Weight']),
            'away_AL': np.average(x['away_AL'], weights=x['Weight']),
            'away_TP': np.average(x['away_TP'], weights=x['Weight']),
            'away_TW': np.average(x['away_TW'], weights=x['Weight']),
            'away_TD': np.average(x['away_TD'], weights=x['Weight']),
            'away_TL': np.average(x['away_TL'], weights=x['Weight']),
            'away_TF': np.average(x['away_TF'], weights=x['Weight']),
            'away_TA': np.average(x['away_TA'], weights=x['Weight']),
            'away_TAG': np.average(x['away_TAG'], weights=x['Weight']),
        })
    ).reset_index()

    # Define features and target
    features = match_results[['HomeTeam', 'AwayTeam', 'home_Pts', 'away_Pts',
                              'home_HP', 'home_HW', 'home_HD', 'home_HL', 'home_AP', 
                              'home_AW', 'home_AD', 'home_AL', 'home_TP', 'home_TW', 
                              'home_TD', 'home_TL', 'home_TF', 'home_TA', 'home_TAG', 
                              'away_HP', 'away_HW', 'away_HD', 'away_HL', 'away_AP', 
                              'away_AW', 'away_AD', 'away_AL', 'away_TP', 'away_TW', 
                              'away_TD', 'away_TL', 'away_TF', 'away_TA', 'away_TAG']]
    
    target = match_results['FTR']

    # Encode team names
    label_encoder_home = LabelEncoder()
    label_encoder_away = LabelEncoder()
    features['HomeTeam'] = label_encoder_home.fit_transform(features['HomeTeam'])
    features['AwayTeam'] = label_encoder_away.fit_transform(features['AwayTeam'])

    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled, target, label_encoder_home, label_encoder_away, scaler, home_stats, away_stats

def build_and_train_model(X_train, y_train):
    """Build and train the deep learning model."""
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))  # Output layer with 3 units for 3 possible outcomes

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

    # Save the model
    model.save('match_prediction_model.h5')
    return model

def predict_match_result(model, scaler, label_encoder_home, label_encoder_away, home_stats, away_stats, home_team, away_team):
    """Predict the match result for given teams."""
    # Encode team names
    home_team_encoded = label_encoder_home.transform([home_team])[0]
    away_team_encoded = label_encoder_away.transform([away_team])[0]

    # Get the mean statistics for the home and away teams
    home_team_mean = home_stats[home_stats['HomeTeam'] == home_team].iloc[0]
    away_team_mean = away_stats[away_stats['AwayTeam'] == away_team].iloc[0]

    # Prepare the feature vector for the prediction
    example_match = pd.DataFrame({
        'HomeTeam': [home_team_encoded],
        'AwayTeam': [away_team_encoded],
        'home_Pts': [home_team_mean['home_Pts']],
        'away_Pts': [away_team_mean['away_Pts']],
        'home_HP': [home_team_mean['home_HP']],
        'home_HW': [home_team_mean['home_HW']],
        'home_HD': [home_team_mean['home_HD']],
        'home_HL': [home_team_mean['home_HL']],
        'home_AP': [home_team_mean['home_AP']],
        'home_AW': [home_team_mean['home_AW']],
        'home_AD': [home_team_mean['home_AD']],
        'home_AL': [home_team_mean['home_AL']],
        'home_TP': [home_team_mean['home_TP']],
        'home_TW': [home_team_mean['home_TW']],
        'home_TD': [home_team_mean['home_TD']],
        'home_TL': [home_team_mean['home_TL']],
        'home_TF': [home_team_mean['home_TF']],
        'home_TA': [home_team_mean['home_TA']],
        'home_TAG': [home_team_mean['home_TAG']],
        'away_HP': [away_team_mean['away_HP']],
        'away_HW': [away_team_mean['away_HW']],
        'away_HD': [away_team_mean['away_HD']],
        'away_HL': [away_team_mean['away_HL']],
        'away_AP': [away_team_mean['away_AP']],
        'away_AW': [away_team_mean['away_AW']],
        'away_AD': [away_team_mean['away_AD']],
        'away_AL': [away_team_mean['away_AL']],
        'away_TP': [away_team_mean['away_TP']],
        'away_TW': [away_team_mean['away_TW']],
        'away_TD': [away_team_mean['away_TD']],
        'away_TL': [away_team_mean['away_TL']],
        'away_TF': [away_team_mean['away_TF']],
        'away_TA': [away_team_mean['away_TA']],
        'away_TAG': [away_team_mean['away_TAG']]
    })

    # Normalize example match data
    example_match_scaled = scaler.transform(example_match)

    # Predict
    predictions = model.predict(example_match_scaled)
    predicted_result = np.argmax(predictions[0])

    # Map result to outcome
    result_map = {0: 'Draw', 1: 'Home Win', 2: 'Away Win'}
    predicted_outcome = result_map[predicted_result]

    return predicted_outcome

# Main workflow
if __name__ == "__main__":
    # Load and prepare data with decay factor
    features_scaled, target, label_encoder_home, label_encoder_away, scaler, home_stats, away_stats = prepare_data('temp.csv', decay_factor=0.1)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

    # Build and train the model
    model = build_and_train_model(X_train, y_train)

    # Example input for prediction
    home_team = 'Chelsea'
    away_team = 'Manchester City'

    # Predict match result
    predicted_outcome = predict_match_result(model, scaler, label_encoder_home, label_encoder_away, home_stats, away_stats, home_team, away_team)
    
    print(f"Predicted result for {home_team} vs. {away_team}: {predicted_outcome}")
