import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.saving import save_model
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
    match_results['Result'] = match_results['Result'].map(ftr_map)

    # Calculate weighted average statistics for home and away teams
    home_stats = match_results.groupby('Home').apply(
        lambda x: pd.Series({
            'Home_Pts': np.average(x['Home_Pts'], weights=x['Weight']),
            'Home_W': np.average(x['Home_W'], weights=x['Weight']),
            'Home_L': np.average(x['Home_L'], weights=x['Weight']),
            'Home_HGD': np.average(x['Home_HGD'], weights=x['Weight']),
            'Home_PPM': np.average(x['Home_PPM'], weights=x['Weight']),
            'Home_HW': np.average(x['Home_HW'], weights=x['Weight']),
            'Home_HL': np.average(x['Home_HL'], weights=x['Weight']),
            'Home_HPts': np.average(x['Home_HPts'], weights=x['Weight']),
            'Home_HPts/MP': np.average(x['Home_HPts/MP'], weights=x['Weight']),
            'Home_Pts/MP': np.average(x['Home_Pts/MP'], weights=x['Weight'])
        })
    ).reset_index()

    away_stats = match_results.groupby('Away').apply(
        lambda x: pd.Series({
            'Away_Pts': np.average(x['Away_Pts'], weights=x['Weight']),
            'Away_W': np.average(x['Away_W'], weights=x['Weight']),
            'Away_L': np.average(x['Away_L'], weights=x['Weight']),
            'Away_AGD': np.average(x['Away_AGD'], weights=x['Weight']),
            'Away_PPM': np.average(x['Away_PPM'], weights=x['Weight']),
            'Away_AW': np.average(x['Away_AW'], weights=x['Weight']),
            'Away_AL': np.average(x['Away_AL'], weights=x['Weight']),
            'Away_APts': np.average(x['Away_APts'], weights=x['Weight']),
            'Away_APts/MP': np.average(x['Away_APts/MP'], weights=x['Weight']),
            'Away_Pts/MP': np.average(x['Away_Pts/MP'], weights=x['Weight'])
        })
    ).reset_index()

    # Define features and target
    feature_columns = ['Home', 'Away', 'Home_Pts', 'Home_W', 'Home_L', 'Home_Pts/MP', 
                       'Home_HW', 'Home_HL', 'Home_HGD', 'Home_HPts', 'Home_HPts/MP', 
                       'Home_PPM', 'Away_Pts', 'Away_W', 'Away_L', 'Away_Pts/MP',
                       'Away_AW', 'Away_AL', 'Away_AGD', 'Away_APts', 
                       'Away_APts/MP', 'Away_PPM']
    
    features = match_results[feature_columns]
    target = match_results['Result']

    # Encode team names
    label_encoder_home = LabelEncoder()
    label_encoder_away = LabelEncoder()
    features['Home'] = label_encoder_home.fit_transform(features['Home'])
    features['Away'] = label_encoder_away.fit_transform(features['Away'])

    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled, target, label_encoder_home, label_encoder_away, scaler, home_stats, away_stats, feature_columns

def apply_pca(features_scaled, n_components=10):
    """Apply PCA to reduce dimensionality."""
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features_scaled)
    return features_pca

def apply_rfe(features, target, n_features_to_select=10):
    """Apply RFE to select important features."""
    model = RandomForestClassifier(n_estimators=100)
    rfe = RFE(model, n_features_to_select=n_features_to_select)
    rfe.fit(features, target)
    return rfe

def build_and_train_model(X_train, y_train):
    """Build and train the deep learning model."""
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))  # Output layer with 3 units for 3 possible outcomes

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

    # Save the model in the recommended Keras format
    save_model(model, 'match_prediction_model.keras')
    return model

def predict_match_result(model, scaler, pca, label_encoder_home, label_encoder_away, home_stats, away_stats, home_team, away_team, feature_columns):
    """Predict the match result for given teams."""
    # Encode team names
    home_team_encoded = label_encoder_home.transform([home_team])[0]
    away_team_encoded = label_encoder_away.transform([away_team])[0]

    # Get the mean statistics for the home and away teams
    home_team_mean = home_stats[home_stats['Home'] == home_team].iloc[0]
    away_team_mean = away_stats[away_stats['Away'] == away_team].iloc[0]

    # Prepare the feature vector for the prediction
    example_match = pd.DataFrame({
        'Home': [home_team_encoded],
        'Away': [away_team_encoded],
        'Home_Pts': [home_team_mean['Home_Pts']],
        'Home_W': [home_team_mean['Home_W']],
        'Home_L': [home_team_mean['Home_L']],
        'Home_HGD': [home_team_mean['Home_HGD']],
        'Home_PPM': [home_team_mean['Home_PPM']],
        'Home_HW': [home_team_mean['Home_HW']],
        'Home_HL': [home_team_mean['Home_HL']],
        'Home_HPts': [home_team_mean['Home_HPts']],
        'Home_HPts/MP': [home_team_mean['Home_HPts/MP']],
        'Home_Pts/MP': [home_team_mean['Home_Pts/MP']],
        'Away_Pts': [away_team_mean['Away_Pts']],
        'Away_W': [away_team_mean['Away_W']],
        'Away_L': [away_team_mean['Away_L']],
        'Away_AGD': [away_team_mean['Away_AGD']],
        'Away_PPM': [away_team_mean['Away_PPM']],
        'Away_AW': [away_team_mean['Away_AW']],
        'Away_AL': [away_team_mean['Away_AL']],
        'Away_APts': [away_team_mean['Away_APts']],
        'Away_APts/MP': [away_team_mean['Away_APts/MP']],
        'Away_Pts/MP': [away_team_mean['Away_Pts/MP']]
    }, columns=feature_columns)

    # Normalize example match data
    example_match_scaled = scaler.transform(example_match)

    # Apply PCA to reduce the example match to the same number of features as the training data
    example_match_pca = pca.transform(example_match_scaled)

    # Predict
    predictions = model.predict(example_match_pca)
    predicted_result = np.argmax(predictions[0])

    # Map result to outcome
    result_map = {0: 'Draw', 1: 'Home Win', 2: 'Away Win'}
    predicted_outcome = result_map[predicted_result]

    return predicted_outcome


# Main workflow
if __name__ == "__main__":
    # Load and prepare data with decay factor
    features_scaled, target, label_encoder_home, label_encoder_away, scaler, home_stats, away_stats, feature_columns = prepare_data('data/combined_fixtures_with_results.csv', decay_factor=0.1)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=10)
    features_pca = pca.fit_transform(features_scaled)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_pca, target, test_size=0.2, random_state=42)
    
    # Build and train the model
    model = build_and_train_model(X_train, y_train)

    # Example input for prediction
    home_team = 'Manchester Utd'
    away_team = 'Liverpool'

    # Predict match result
    predicted_outcome = predict_match_result(
        model, 
        scaler, 
        pca, 
        label_encoder_home, 
        label_encoder_away, 
        home_stats, 
        away_stats, 
        home_team, 
        away_team, 
        feature_columns  # This argument is essential
    )
    
    print(f"Predicted result for {home_team} vs. {away_team}: {predicted_outcome}")

