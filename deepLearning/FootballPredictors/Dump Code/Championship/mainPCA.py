import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split,KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dropout
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

    ftr_map = {'H':1, 'D':0, 'A':2}
    match_result['Result'] = match_result['Result'].map(ftr_map)

    #The reason that it is needed to use two instances of encoding is so that there are two instances one for home and one for away. If there was just one it would conflict as one team would have two different encoding, one in home and one in away.
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

    features = match_result[[
        'Wk', 'Home', 'Away'] + home_columns + away_columns
    ]

    target = match_result['Result']

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
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

    return model

def cross_validate_model(features, target, k=5):

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies=[]

    for train_index, test_index in kf.split(features):
        X_train, X_test = features[train_index], features[test_index]
        Y_train, Y_test = target[train_index], target[test_index]

        model = build_and_train_model(X_train, Y_train)
        model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=50, batch_size=32, verbose=0)

        accuracy = model.evaluate(X_test, Y_test, verbose=0)
        accuracies.append(accuracy)
    
    return np.mean(accuracies)

def predict_match_result(model, scalar, pca, label_encoder_home, label_encoder_away, home_stats, away_stats, home_team, away_team):

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

    probabilities = model.predict(example_match_pca)[0]
    home_win_prob = probabilities[1]
    draw_prob = probabilities[0]
    away_win_prob = probabilities[2]
    outcome_labels = ['Home Win', 'Draw', 'Away Win']
    predicted_probabilities = [home_win_prob, draw_prob, away_win_prob]

    return outcome_labels, predicted_probabilities


if __name__ == "__main__":

    features_pca, target, label_encoder_home, label_encoder_away, scalar, pca, home_stats, away_stats = prepare_data('data/final.csv', decay_factor=0.1, n_components=50)

    avg_accuracy = cross_validate_model(features_pca, target, k=5)
    print(f'Average cross-validated accuracy: {avg_accuracy:.2f}')

    X_train, X_test, y_train, y_test = train_test_split(features_pca, target, test_size=0.2, random_state=42)

    model = build_and_train_model(X_train, y_train)
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1) 

    home_teams = ['Oxford United']
    away_teams = ['West Brom']
    wk = '10' 

    for home_team, away_team in zip(home_teams, away_teams):
        outcome_labels, predicted_probabilities = predict_match_result(model, scalar, pca, label_encoder_home, label_encoder_away, home_stats, away_stats, home_team, away_team)

        print(f"Predicted probabilities for {home_team} vs. {away_team}:")
        for label, prob in zip(outcome_labels, predicted_probabilities):
            print(f"{label}: {prob:.2f}")
