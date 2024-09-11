import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
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

    # Calculate weighted average statistics for home and away teams
    # Update home stats calculation

    home_stats = match_results.groupby('Home').apply(
        lambda x: pd.Series({
            'xG': np.average(x['xG'], weights=x['Weight']),
            'Home_HW': np.average(x['Home_HW'], weights=x['Weight']),
            'Home_HPts/MP': np.average(x['Home_HPts/MP'], weights=x['Weight']),
            'Home_HPts': np.average(x['Home_HPts'], weights=x['Weight']),
            'Home_HGD': np.average(x['Home_HGD'], weights=x['Weight']),
            'Home_HL': np.average(x['Home_HL'], weights=x['Weight']),
            'Home_Gls': np.average(x['Home_Gls'], weights=x['Weight']),
            'Home_+/-90': np.average(x['Home_+/-90'], weights=x['Weight']),
        })
    ).reset_index()

    away_stats = match_results.groupby('Away').apply(
        lambda x: pd.Series({
            'xG.1': np.average(x['xG.1'], weights=x['Weight']),
            'Away_APts/MP': np.average(x['Away_APts/MP'], weights=x['Weight']),
            'Away_AGD': np.average(x['Away_AGD'], weights=x['Weight']),
            'Away_AL': np.average(x['Away_AL'], weights=x['Weight']),
            'Away_APts': np.average(x['Away_APts'], weights=x['Weight']),
            'Away_GA90': np.average(x['Away_GA90'], weights=x['Weight']),
            'Away_AW': np.average(x['Away_AW'], weights=x['Weight']),
            'Away_L': np.average(x['Away_L'], weights=x['Weight']),
            'Away_Pts': np.average(x['Away_Pts'], weights=x['Weight']),
            'Away_PPM': np.average(x['Away_PPM'], weights=x['Weight']),
            'Away_Pts/MP': np.average(x['Away_Pts/MP'], weights=x['Weight']),
        })
    ).reset_index()



    features = match_results[['Home','Away', 'Wk', 'xG', 'Home_HW', 'Home_HPts/MP', 
                            'Home_HPts','Home_HGD', 'Home_HL','Home_Gls','Home_+/-90', 'xG.1',
                            'Away_APts/MP', 'Away_AGD', 'Away_AL', 'Away_APts', 'Away_GA90', 'Away_AW', 
                            'Away_L', 'Away_Pts', 'Away_PPM', 'Away_Pts/MP']]
        
    target = match_results['Result']


    # Encode team names
    label_encoder_home = LabelEncoder()
    label_encoder_away = LabelEncoder()
    features['Home'] = label_encoder_home.fit_transform(features['Home'])
    features['Away'] = label_encoder_away.fit_transform(features['Away'])

    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled, target, label_encoder_home, label_encoder_away, scaler, home_stats, away_stats

def build_and_train_model(X_train, y_train):
    """Build and train the deep learning model."""
    model = Sequential()
    model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))  # Output layer with 3 units for 3 possible outcomes

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.1)

    # Save the model
    model.save('match_prediction_model.h5')
    return model

def predict_match_result(model, scaler, label_encoder_home, label_encoder_away, home_stats, away_stats, home_team, away_team):
    """Predict the match result for given teams and display the probability of the predicted outcome."""
    # Encode team names
    home_team_encoded = label_encoder_home.transform([home_team])[0]
    away_team_encoded = label_encoder_away.transform([away_team])[0]


    example_match = pd.DataFrame({
        'Home': [home_team_encoded],
        'Away': [away_team_encoded],
        'Wk': [wk],
        'xG': [home_stats.loc[home_team_encoded, 'xG']],
        'Home_HW': [home_stats.loc[home_team_encoded, 'Home_HW']],
        'Home_HPts/MP': [home_stats.loc[home_team_encoded, 'Home_HPts/MP']],
        'Home_HPts': [home_stats.loc[home_team_encoded, 'Home_HPts']],
        'Home_HGD': [home_stats.loc[home_team_encoded, 'Home_HGD']],
        'Home_HL': [home_stats.loc[home_team_encoded, 'Home_HL']],
        'Home_Gls': [home_stats.loc[home_team_encoded, 'Home_Gls']],
        'Home_+/-90': [home_stats.loc[home_team_encoded, 'Home_+/-90']],
        'xG.1': [away_stats.loc[away_team_encoded, 'xG.1']],
        'Away_APts/MP': [away_stats.loc[away_team_encoded, 'Away_APts/MP']],
        'Away_AGD': [away_stats.loc[away_team_encoded, 'Away_AGD']],
        'Away_AL': [away_stats.loc[away_team_encoded, 'Away_AL']],
        'Away_APts': [away_stats.loc[away_team_encoded, 'Away_APts']],
        'Away_GA90': [away_stats.loc[away_team_encoded, 'Away_GA90']],
        'Away_AW': [away_stats.loc[away_team_encoded, 'Away_AW']],
        'Away_L': [away_stats.loc[away_team_encoded, 'Away_L']],
        'Away_Pts': [away_stats.loc[away_team_encoded, 'Away_Pts']],
        'Away_PPM': [away_stats.loc[away_team_encoded, 'Away_PPM']],
        'Away_Pts/MP': [away_stats.loc[away_team_encoded, 'Away_Pts/MP']]
    })

    # Normalize example match data
    example_match_scaled = scaler.transform(example_match)

    # Predict
    predictions = model.predict(example_match_scaled)
    predicted_result = np.argmax(predictions[0])
    predicted_probability = predictions[0][predicted_result]  # Probability of the predicted outcome

    # Map result to outcome
    result_map = {0: 'Draw', 1: 'Home Win', 2: 'Away Win'}
    predicted_outcome = result_map[predicted_result]

    return predicted_outcome, predicted_probability

# Main workflow
if __name__ == "__main__":
    # Load and prepare data with decay factor
    features_scaled, target, label_encoder_home, label_encoder_away, scaler, home_stats, away_stats = prepare_data('data.csv', decay_factor=0.1)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

    # Build and train the model
    model = build_and_train_model(X_train, y_train)

    # Example input for prediction
    home_team = 'Southampton'
    away_team = 'Manchester Utd'
    wk = '4'

    # Predict match result

    predicted_outcome, predicted_probability = predict_match_result(model, scaler, label_encoder_home, label_encoder_away, home_stats, away_stats, home_team, away_team)
    
    print(f"Predicted result for {home_team} vs. {away_team}: {predicted_outcome} with probability {predicted_probability:.2f}")
