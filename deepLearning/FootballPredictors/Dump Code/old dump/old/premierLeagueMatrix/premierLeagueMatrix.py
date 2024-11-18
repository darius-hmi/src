import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

def prepare_data(file_path):
    """Load and prepare the dataset."""
    match_results = pd.read_csv(file_path)

    # Check for non-numeric values in FTR and features
    print("Unique values in FTR:", match_results['FTR'].unique())
    print("Data types in dataset:", match_results.dtypes)

    # Map FTR values to numeric (Home win: 1, Draw: 0, Away win: 2)
    ftr_map = {'H': 1, 'D': 0, 'A': 2}
    match_results['FTR'] = match_results['FTR'].map(ftr_map)

    # Calculate average statistics for home and away teams
    home_stats = match_results.groupby('HomeTeam').agg({
        'home_Pts': 'mean', 'home_HP': 'mean', 'home_HW': 'mean', 'home_HD': 'mean',
        'home_HL': 'mean', 'home_AP': 'mean', 'home_AW': 'mean', 'home_AD': 'mean',
        'home_AL': 'mean', 'home_TP': 'mean', 'home_TW': 'mean', 'home_TD': 'mean',
        'home_TL': 'mean', 'home_TF': 'mean', 'home_TA': 'mean', 'home_TAG': 'mean'
    }).reset_index()
    
    away_stats = match_results.groupby('AwayTeam').agg({
        'away_Pts': 'mean', 'away_HP': 'mean', 'away_HW': 'mean', 'away_HD': 'mean',
        'away_HL': 'mean', 'away_AP': 'mean', 'away_AW': 'mean', 'away_AD': 'mean',
        'away_AL': 'mean', 'away_TP': 'mean', 'away_TW': 'mean', 'away_TD': 'mean',
        'away_TL': 'mean', 'away_TF': 'mean', 'away_TA': 'mean', 'away_TAG': 'mean'
    }).reset_index()

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
    # Load and prepare data
    features_scaled, target, label_encoder_home, label_encoder_away, scaler, home_stats, away_stats = prepare_data('temp.csv')

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

    # Build and train the model
    model = build_and_train_model(X_train, y_train)

    # Example input for prediction
    home_team = 'Everton'
    away_team = 'Manchester City'

    # Predict match result
    predicted_outcome = predict_match_result(model, scaler, label_encoder_home, label_encoder_away, home_stats, away_stats, home_team, away_team)
    
    print(f"Predicted result for {home_team} vs. {away_team}: {predicted_outcome}")
