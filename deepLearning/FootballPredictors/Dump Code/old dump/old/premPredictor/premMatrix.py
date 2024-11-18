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
    match_results['Result'] = match_results['Result'].map(ftr_map)

    # Calculate weighted average statistics for home and away teams
    # Update home stats calculation
    home_stats = match_results.groupby('Home').apply(
        lambda x: pd.Series({
            'Home_Pts': np.average(x['Home_Pts'], weights=x['Weight']),
            'Home_W': np.average(x['Home_W'], weights=x['Weight']),
            'Home_L': np.average(x['Home_L'], weights=x['Weight']),
            'Home_GF': np.average(x['Home_GF'], weights=x['Weight']),
            'Home_GA': np.average(x['Home_GA'], weights=x['Weight']),
            'Home_xG': np.average(x['Home_xG'], weights=x['Weight']),
            'Home_xGD/90': np.average(x['Home_xGD/90'], weights=x['Weight']),
            'Home_Cmp': np.average(x['Home_Cmp'], weights=x['Weight']),
            'Home_Att': np.average(x['Home_Att'], weights=x['Weight']),
            'Home_Ast': np.average(x['Home_Ast'], weights=x['Weight']),
            'Home_KP': np.average(x['Home_KP'], weights=x['Weight']),
            'Home_PPA': np.average(x['Home_PPA'], weights=x['Weight']),
            'Home_Live': np.average(x['Home_Live'], weights=x['Weight']),
            'Home_FK': np.average(x['Home_FK'], weights=x['Weight']),
            'Home_Crs': np.average(x['Home_Crs'], weights=x['Weight']),
            'Home_Blocks': np.average(x['Home_Blocks'], weights=x['Weight']),
            'Home_PassLive': np.average(x['Home_PassLive'], weights=x['Weight']),
            'Home_Att 3rd': np.average(x['Home_Att 3rd'], weights=x['Weight']),
            'Home_Lost': np.average(x['Home_Lost'], weights=x['Weight']),
            'Home_Poss': np.average(x['Home_Poss'], weights=x['Weight']),
            'Home_Touches': np.average(x['Home_Touches'], weights=x['Weight']),
            'Home_Carries': np.average(x['Home_Carries'], weights=x['Weight']),
            'Home_PrgC': np.average(x['Home_PrgC'], weights=x['Weight']),
            'Home_CPA': np.average(x['Home_CPA'], weights=x['Weight']),
            'Home_HGF': np.average(x['Home_HGF'], weights=x['Weight']),
            'Home_HGD': np.average(x['Home_HGD'], weights=x['Weight']),
            'Home_AW': np.average(x['Home_AW'], weights=x['Weight']),
            'Home_Recov': np.average(x['Home_Recov'], weights=x['Weight']),
            'Home_Saves': np.average(x['Home_Saves'], weights=x['Weight']),
            'Home_SoT/90': np.average(x['Home_SoT/90'], weights=x['Weight']),
            'Home_G/SoT': np.average(x['Home_G/SoT'], weights=x['Weight']),
            'Home_Dist': np.average(x['Home_Dist'], weights=x['Weight'])
        })
    ).reset_index()

    # Update away stats calculation
    away_stats = match_results.groupby('Away').apply(
        lambda x: pd.Series({
            'Away_Pts': np.average(x['Away_Pts'], weights=x['Weight']),
            'Away_W': np.average(x['Away_W'], weights=x['Weight']),
            'Away_L': np.average(x['Away_L'], weights=x['Weight']),
            'Away_GF': np.average(x['Away_GF'], weights=x['Weight']),
            'Away_GA': np.average(x['Away_GA'], weights=x['Weight']),
            'Away_xG': np.average(x['Away_xG'], weights=x['Weight']),
            'Away_xGD/90': np.average(x['Away_xGD/90'], weights=x['Weight']),
            'Away_Cmp': np.average(x['Away_Cmp'], weights=x['Weight']),
            'Away_Att': np.average(x['Away_Att'], weights=x['Weight']),
            'Away_Ast': np.average(x['Away_Ast'], weights=x['Weight']),
            'Away_KP': np.average(x['Away_KP'], weights=x['Weight']),
            'Away_PPA': np.average(x['Away_PPA'], weights=x['Weight']),
            'Away_Live': np.average(x['Away_Live'], weights=x['Weight']),
            'Away_FK': np.average(x['Away_FK'], weights=x['Weight']),
            'Away_Crs': np.average(x['Away_Crs'], weights=x['Weight']),
            'Away_Blocks': np.average(x['Away_Blocks'], weights=x['Weight']),
            'Away_PassLive': np.average(x['Away_PassLive'], weights=x['Weight']),
            'Away_Att 3rd': np.average(x['Away_Att 3rd'], weights=x['Weight']),
            'Away_Lost': np.average(x['Away_Lost'], weights=x['Weight']),
            'Away_Poss': np.average(x['Away_Poss'], weights=x['Weight']),
            'Away_Touches': np.average(x['Away_Touches'], weights=x['Weight']),
            'Away_Carries': np.average(x['Away_Carries'], weights=x['Weight']),
            'Away_PrgC': np.average(x['Away_PrgC'], weights=x['Weight']),
            'Away_CPA': np.average(x['Away_CPA'], weights=x['Weight']),
            'Away_HGF': np.average(x['Away_HGF'], weights=x['Weight']),
            'Away_HGD': np.average(x['Away_HGD'], weights=x['Weight']),
            'Away_AW': np.average(x['Away_AW'], weights=x['Weight']),
            'Away_Recov': np.average(x['Away_Recov'], weights=x['Weight']),
            'Away_Saves': np.average(x['Away_Saves'], weights=x['Weight']),
            'Away_SoT/90': np.average(x['Away_SoT/90'], weights=x['Weight']),
            'Away_G/SoT': np.average(x['Away_G/SoT'], weights=x['Weight']),
            'Away_Dist': np.average(x['Away_Dist'], weights=x['Weight'])
        })
    ).reset_index()


    # Define features and target
    features = match_results[['Home', 'Away', 'Home_Pts', 'Home_W',
                              'Home_L', 'Home_GF', 'Home_GA', 'Home_xG', 'Home_xGD/90', 
                              'Home_Cmp', 'Home_Att', 'Home_Ast', 'Home_KP', 'Home_PPA', 
                              'Home_Live', 'Home_FK', 'Home_Crs', 'Home_Blocks', 'Home_PassLive', 
                              'Home_Att 3rd', 'Home_Lost', 'Home_Poss', 'Home_Touches', 'Home_Carries', 
                              'Home_PrgC', 'Home_CPA', 'Home_HGF', 'Home_HGD', 'Home_AW', 
                              'Home_Recov', 'Home_Saves', 'Home_SoT/90', 'Home_G/SoT', 'Home_Dist',
                              'Away_Pts', 'Away_W','Away_L', 'Away_GF', 'Away_GA', 'Away_xG', 'Away_xGD/90', 
                              'Away_Cmp', 'Away_Att', 'Away_Ast', 'Away_KP', 'Away_PPA', 
                              'Away_Live', 'Away_FK', 'Away_Crs', 'Away_Blocks', 'Away_PassLive', 
                              'Away_Att 3rd', 'Away_Lost', 'Away_Poss', 'Away_Touches', 'Away_Carries', 
                              'Away_PrgC', 'Away_CPA', 'Away_HGF', 'Away_HGD', 'Away_AW', 
                              'Away_Recov', 'Away_Saves', 'Away_SoT/90', 'Away_G/SoT', 'Away_Dist']]
    
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
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))  # Output layer with 3 units for 3 possible outcomes

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

    # Save the model
    model.save('match_prediction_model.h5')
    return model

def predict_match_result(model, scaler, label_encoder_home, label_encoder_away, home_stats, away_stats, home_team, away_team):
    """Predict the match result for given teams and display the probability of the predicted outcome."""
    # Encode team names
    home_team_encoded = label_encoder_home.transform([home_team])[0]
    away_team_encoded = label_encoder_away.transform([away_team])[0]

    # Prepare the feature vector for the prediction
    example_match = pd.DataFrame({
        'Home': [home_team_encoded],
        'Away': [away_team_encoded],
        'Home_Pts': [home_stats.loc[home_team_encoded, 'Home_Pts']],
        'Home_W': [home_stats.loc[home_team_encoded, 'Home_W']],
        'Home_L': [home_stats.loc[home_team_encoded, 'Home_L']],
        'Home_GF': [home_stats.loc[home_team_encoded, 'Home_GF']],
        'Home_GA': [home_stats.loc[home_team_encoded, 'Home_GA']],
        'Home_xG': [home_stats.loc[home_team_encoded, 'Home_xG']],
        'Home_xGD/90': [home_stats.loc[home_team_encoded, 'Home_xGD/90']],
        'Home_Cmp': [home_stats.loc[home_team_encoded, 'Home_Cmp']],
        'Home_Att': [home_stats.loc[home_team_encoded, 'Home_Att']],
        'Home_Ast': [home_stats.loc[home_team_encoded, 'Home_Ast']],
        'Home_KP': [home_stats.loc[home_team_encoded, 'Home_KP']],
        'Home_PPA': [home_stats.loc[home_team_encoded, 'Home_PPA']],
        'Home_Live': [home_stats.loc[home_team_encoded, 'Home_Live']],
        'Home_FK': [home_stats.loc[home_team_encoded, 'Home_FK']],
        'Home_Crs': [home_stats.loc[home_team_encoded, 'Home_Crs']],
        'Home_Blocks': [home_stats.loc[home_team_encoded, 'Home_Blocks']],
        'Home_PassLive': [home_stats.loc[home_team_encoded, 'Home_PassLive']],
        'Home_Att 3rd': [home_stats.loc[home_team_encoded, 'Home_Att 3rd']],
        'Home_Lost': [home_stats.loc[home_team_encoded, 'Home_Lost']],
        'Home_Poss': [home_stats.loc[home_team_encoded, 'Home_Poss']],
        'Home_Touches': [home_stats.loc[home_team_encoded, 'Home_Touches']],
        'Home_Carries': [home_stats.loc[home_team_encoded, 'Home_Carries']],
        'Home_PrgC': [home_stats.loc[home_team_encoded, 'Home_PrgC']],
        'Home_CPA': [home_stats.loc[home_team_encoded, 'Home_CPA']],
        'Home_HGF': [home_stats.loc[home_team_encoded, 'Home_HGF']],
        'Home_HGD': [home_stats.loc[home_team_encoded, 'Home_HGD']],
        'Home_AW': [home_stats.loc[home_team_encoded, 'Home_AW']],
        'Home_Recov': [home_stats.loc[home_team_encoded, 'Home_Recov']],
        'Home_Saves': [home_stats.loc[home_team_encoded, 'Home_Saves']],
        'Home_SoT/90': [home_stats.loc[home_team_encoded, 'Home_SoT/90']],
        'Home_G/SoT': [home_stats.loc[home_team_encoded, 'Home_G/SoT']],
        'Home_Dist': [home_stats.loc[home_team_encoded, 'Home_Dist']],
        'Away_Pts': [away_stats.loc[away_team_encoded, 'Away_Pts']],
        'Away_W': [away_stats.loc[away_team_encoded, 'Away_W']],
        'Away_L': [away_stats.loc[away_team_encoded, 'Away_L']],
        'Away_GF': [away_stats.loc[away_team_encoded, 'Away_GF']],
        'Away_GA': [away_stats.loc[away_team_encoded, 'Away_GA']],
        'Away_xG': [away_stats.loc[away_team_encoded, 'Away_xG']],
        'Away_xGD/90': [away_stats.loc[away_team_encoded, 'Away_xGD/90']],
        'Away_Cmp': [away_stats.loc[away_team_encoded, 'Away_Cmp']],
        'Away_Att': [away_stats.loc[away_team_encoded, 'Away_Att']],
        'Away_Ast': [away_stats.loc[away_team_encoded, 'Away_Ast']],
        'Away_KP': [away_stats.loc[away_team_encoded, 'Away_KP']],
        'Away_PPA': [away_stats.loc[away_team_encoded, 'Away_PPA']],
        'Away_Live': [away_stats.loc[away_team_encoded, 'Away_Live']],
        'Away_FK': [away_stats.loc[away_team_encoded, 'Away_FK']],
        'Away_Crs': [away_stats.loc[away_team_encoded, 'Away_Crs']],
        'Away_Blocks': [away_stats.loc[away_team_encoded, 'Away_Blocks']],
        'Away_PassLive': [away_stats.loc[away_team_encoded, 'Away_PassLive']],
        'Away_Att 3rd': [away_stats.loc[away_team_encoded, 'Away_Att 3rd']],
        'Away_Lost': [away_stats.loc[away_team_encoded, 'Away_Lost']],
        'Away_Poss': [away_stats.loc[away_team_encoded, 'Away_Poss']],
        'Away_Touches': [away_stats.loc[away_team_encoded, 'Away_Touches']],
        'Away_Carries': [away_stats.loc[away_team_encoded, 'Away_Carries']],
        'Away_PrgC': [away_stats.loc[away_team_encoded, 'Away_PrgC']],
        'Away_CPA': [away_stats.loc[away_team_encoded, 'Away_CPA']],
        'Away_HGF': [away_stats.loc[away_team_encoded, 'Away_HGF']],
        'Away_HGD': [away_stats.loc[away_team_encoded, 'Away_HGD']],
        'Away_AW': [away_stats.loc[away_team_encoded, 'Away_AW']],
        'Away_Recov': [away_stats.loc[away_team_encoded, 'Away_Recov']],
        'Away_Saves': [away_stats.loc[away_team_encoded, 'Away_Saves']],
        'Away_SoT/90': [away_stats.loc[away_team_encoded, 'Away_SoT/90']],
        'Away_G/SoT': [away_stats.loc[away_team_encoded, 'Away_G/SoT']],
        'Away_Dist': [away_stats.loc[away_team_encoded, 'Away_Dist']]
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
    features_scaled, target, label_encoder_home, label_encoder_away, scaler, home_stats, away_stats = prepare_data('data/combined_fixtures_with_results.csv', decay_factor=0.1)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

    # Build and train the model
    model = build_and_train_model(X_train, y_train)

    # Example input for prediction
    home_team = 'Southampton'
    away_team = 'Manchester Utd'

    # Predict match result

    predicted_outcome, predicted_probability = predict_match_result(model, scaler, label_encoder_home, label_encoder_away, home_stats, away_stats, home_team, away_team)
    
    print(f"Predicted result for {home_team} vs. {away_team}: {predicted_outcome} with probability {predicted_probability:.2f}")
