import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping



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

    # Extract columns related to away stats and add xG.1
    away_columns = [col for col in match_results.columns if col.startswith(away_prefix)] + additional_away_columns


    def calculate_weighted_averages(df, group_by_col, columns, weight_col='Weight'):
        return df.groupby(group_by_col).apply(
            lambda x: pd.Series({
                col: np.average(x[col], weights=x[weight_col]) if col in x.columns else np.nan
                for col in columns
            })
        ).reset_index()

    home_stats = calculate_weighted_averages(match_results, 'Home', home_columns)

    away_stats = calculate_weighted_averages(match_results, 'Away', away_columns)

    features = match_results[[
        'Wk', 'Home', 'Away'] + home_columns + away_columns
    ]
        
    target = match_results['Result']


    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled, target, label_encoder_home, label_encoder_away, scaler, home_stats, away_stats

def build_and_train_model(X_train, y_train):
    """Build and train the deep learning model."""
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.2)))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.2)))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax'))  # Output layer with 3 units for 3 possible outcomes

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
    # Save the model
    return model

def cross_validate_model(features, target, k=5):
    """Perform K-Fold cross-validation and return average accuracy."""
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []

    for train_index, test_index in kf.split(features):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = target[train_index], target[test_index]

        model = build_and_train_model(X_train, y_train)
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        accuracies.append(accuracy)

    return np.mean(accuracies)

def predict_match_result(model, scaler, label_encoder_home, label_encoder_away, home_stats, away_stats, home_team, away_team):
    """Predict the match result for given teams and display the probability of the predicted outcome."""
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

    # Predict
    probabilities = model.predict(example_match_scaled)[0]  # Model returns an array of probabilities
    
    # Decode probabilities
    home_win_prob = probabilities[1]
    draw_prob = probabilities[0]
    away_win_prob = probabilities[2]

    # Create labels for the outcomes
    outcome_labels = ['Home Win', 'Draw', 'Away Win']
    predicted_probabilities = [home_win_prob, draw_prob, away_win_prob]

    return outcome_labels, predicted_probabilities

# Main workflow
if __name__ == "__main__":
    # Load and prepare data with decay factor
    features_scaled, target, label_encoder_home, label_encoder_away, scaler, home_stats, away_stats = prepare_data('data/final45_newFeatures.csv', decay_factor=0.1)

    average_accuracy = cross_validate_model(features_scaled, target, k=5)
    print(f'Average Accuracy from Cross-Validation: {average_accuracy:.2f}')
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

    # Build and train the model
    model = build_and_train_model(X_train, y_train)

    # Example input for prediction
    home_teams = ['Brentford', 'Crystal Palace', 'West Ham', 'Wolves', 'Brighton', 'Liverpool', 'Manchester Utd', 'Nott\'ham Forest', 'Tottenham', 'Chelsea']
    away_teams = ['Bournemouth', 'Fulham', 'Everton', 'Southampton', 'Manchester City', 'Aston Villa', 'Leicester City', 'Newcastle Utd', 'Ipswich Town', 'Arsenal'] 
    wk = '11' 

    for home_team, away_team in zip(home_teams, away_teams):
        outcome_labels, predicted_probabilities = predict_match_result(model, scaler, label_encoder_home, label_encoder_away, home_stats, away_stats, home_team, away_team)
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
        print(f"Final Test Loss: {test_loss:.4f}")
        print(f"Final Test Accuracy: {test_accuracy:.4f}")

        print(f"Predicted probabilities for {home_team} vs. {away_team}:")
        for label, prob in zip(outcome_labels, predicted_probabilities):
            print(f"{label}: {prob:.2f}")

    


