import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.saving import save_model
import numpy as np

def exponential_decay_weight(df, decay_factor):
    """Apply exponential decay to the season weights."""
    max_season = df['Season'].max()
    weights = np.exp(-decay_factor * (max_season - df['Season']))
    return weights

def prepare_data(file_path, decay_factor=0.1, n_components=20):
    """Load and prepare the dataset with time decay and apply PCA."""
    match_results = pd.read_csv(file_path)
    match_results['Season'] = pd.to_datetime(match_results['Date']).dt.year

    # Apply exponential decay to the weights
    match_results['Weight'] = exponential_decay_weight(match_results, decay_factor)

    # Map FTR values to numeric (Home win: 1, Draw: 0, Away win: 2)
    ftr_map = {'H': 1, 'D': 0, 'A': 2}
    match_results['Result'] = match_results['Result'].map(ftr_map)

    # Define features and target
    feature_columns = match_results.columns.drop(['Result', 'Date', 'Season', 'Weight', 'Day', 'Time', 'Score', 'Referee', 'Home_Weekly Wages', 'Home_Annual Wages', 'Home_% Estimated','Away_Weekly Wages', 'Away_Annual Wages', 'Away_% Estimated']).tolist()
    features = match_results[feature_columns]
    target = match_results['Result']

    # Encode categorical columns ('Home' and 'Away')
    label_encoder_home = LabelEncoder()
    label_encoder_away = LabelEncoder()
    features['Home'] = label_encoder_home.fit_transform(features['Home'])
    features['Away'] = label_encoder_away.fit_transform(features['Away'])

    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features_scaled)

    return features_pca, target, label_encoder_home, label_encoder_away, scaler, pca, feature_columns, features.shape[1]

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

def predict_match_result(model, scaler, pca, label_encoder_home, label_encoder_away, home_team, away_team, original_feature_count):
    """Predict the match result for given teams with all necessary features."""
    # Encode team names
    home_team_encoded = label_encoder_home.transform([home_team])[0]
    away_team_encoded = label_encoder_away.transform([away_team])[0]

    # Prepare a base feature vector filled with zeros
    feature_vector = np.zeros((1, original_feature_count))
    
    # Place the 'Home' and 'Away' encoded values in their correct positions (5 and 10)
    feature_vector[0, 5] = home_team_encoded
    feature_vector[0, 10] = away_team_encoded
    
    # Normalize the feature vector
    feature_vector_scaled = scaler.transform(feature_vector)
    
    # Apply PCA to the feature vector
    feature_vector_pca = pca.transform(feature_vector_scaled)
    
    # Predict
    predictions = model.predict(feature_vector_pca)
    predicted_result = np.argmax(predictions[0])

    # Map result to outcome
    result_map = {0: 'Draw', 1: 'Home Win', 2: 'Away Win'}
    predicted_outcome = result_map[predicted_result]

    return predicted_outcome

# Main workflow
if __name__ == "__main__":
    # Load and prepare data with decay factor and PCA
    features_pca, target, label_encoder_home, label_encoder_away, scaler, pca, feature_columns, original_feature_count = prepare_data(
        'data/combined_fixtures_with_results.csv', decay_factor=0.1, n_components=20
    )

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_pca, target, test_size=0.2, random_state=42)

    # Build and train the model
    model = build_and_train_model(X_train, y_train)

    # Example input for prediction
    home_team = 'Wolves'
    away_team = 'Chelsea'

    # Predict match result
    predicted_outcome = predict_match_result(
        model, scaler, pca, label_encoder_home, label_encoder_away,
        home_team, away_team, original_feature_count
    )
    
    print(f"Predicted result for {home_team} vs. {away_team}: {predicted_outcome}")
