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

    # Calculate weighted averages for home stats including xG
    home_stats = calculate_weighted_averages(match_results, 'Home', home_columns)

    # Calculate weighted averages for away stats including xG.1
    away_stats = calculate_weighted_averages(match_results, 'Away', away_columns)
    # Calculate weighted average statistics for home and away teams
    # Update home stats calculation

    # home_stats = match_results.groupby('Home').apply(
    #     lambda x: pd.Series({
    #         'xG': np.average(x['xG'], weights=x['Weight']),
    #         'Home_HW': np.average(x['Home_HW'], weights=x['Weight']),
    #         'Home_HPts/MP': np.average(x['Home_HPts/MP'], weights=x['Weight']),
    #         'Home_HPts': np.average(x['Home_HPts'], weights=x['Weight']),
    #         'Home_HGD': np.average(x['Home_HGD'], weights=x['Weight']),
    #         'Home_HL': np.average(x['Home_HL'], weights=x['Weight']),
    #         'Home_Gls': np.average(x['Home_Gls'], weights=x['Weight']),
    #         'Home_+/-90': np.average(x['Home_+/-90'], weights=x['Weight']),
    #     })
    # ).reset_index()

    # away_stats = match_results.groupby('Away').apply(
    #     lambda x: pd.Series({
    #         'xG.1': np.average(x['xG.1'], weights=x['Weight']),
    #         'Away_APts/MP': np.average(x['Away_APts/MP'], weights=x['Weight']),
    #         'Away_AGD': np.average(x['Away_AGD'], weights=x['Weight']),
    #         'Away_AL': np.average(x['Away_AL'], weights=x['Weight']),
    #         'Away_APts': np.average(x['Away_APts'], weights=x['Weight']),
    #         'Away_GA90': np.average(x['Away_GA90'], weights=x['Weight']),
    #         'Away_AW': np.average(x['Away_AW'], weights=x['Weight']),
    #         'Away_L': np.average(x['Away_L'], weights=x['Weight']),
    #         'Away_Pts': np.average(x['Away_Pts'], weights=x['Weight']),
    #         'Away_PPM': np.average(x['Away_PPM'], weights=x['Weight']),
    #         'Away_Pts/MP': np.average(x['Away_Pts/MP'], weights=x['Weight']),
    #     })
    # ).reset_index()



    features = match_results[[
        'Wk', 'Home', 'Away', 'Home_Rk', 'Home_MP', 'Home_W', 'Home_D', 'Home_L',
        'Home_GF', 'Home_GA', 'Home_GD', 'Home_Pts', 'Home_Pts/MP', 'Home_xG', 'Home_xGA', 'Home_xGD', 'Home_xGD/90',
        'Home_HMP', 'Home_HW', 'Home_HD', 'Home_HL', 'Home_HGF', 'Home_HGA', 'Home_HGD', 'Home_HPts', 'Home_HPts/MP',
        'Home_HxG', 'Home_HxGA', 'Home_HxGD', 'Home_AxGD/90', 'Home_AxGD/90_1', 'Home_AMP', 'Home_AW', 'Home_AD',
        'Home_AL', 'Home_AGF', 'Home_AGA', 'Home_AGD', 'Home_APts', 'Home_APts/MP', 'Home_AxG', 'Home_AxGA',
        'Home_AxGD', 'Home_AxGD/90_2', 'Home_AxGD/90_3', 'Home_Age', 'Home_Poss', 'Home_Ast', 'Home_Ast_1', 'Home_G+A',
        'Home_G+A_1', 'Home_G-PK', 'Home_G-PK_1', 'Home_PK', 'Home_PKatt', 'Home_CrdY', 'Home_CrdR', 'Home_npxG',
        'Home_npxG_1', 'Home_xAG', 'Home_xAG_1', 'Home_npxG+xAG', 'Home_npxG+xAG_1', 'Home_PrgC', 'Home_PrgP',
        'Home_Ast_2', 'Home_Ast_3', 'Home_G+A_2', 'Home_G+A_3', 'Home_G-PK_2', 'Home_G-PK_3', 'Home_G+A-PK',
        'Home_xAG_2', 'Home_xAG_3', 'Home_xG+xAG', 'Home_npxG_2', 'Home_npxG_3', 'Home_npxG+xAG_2', 'Home_npxG+xAG_3',
        'Home_GA90', 'Home_SoTA', 'Home_Saves', 'Home_Save%', 'Home_CS', 'Home_CS%', 'Home_PKA', 'Home_PKsv', 'Home_PKm',
        'Home_Save%_1', 'Home_FK', 'Home_CK', 'Home_OG', 'Home_PSxG', 'Home_PSxG/SoT', 'Home_PSxG+/-', 'Home_/90',
        'Home_Cmp', 'Home_Att', 'Home_Att_1', 'Home_Cmp%', 'Home_Att (GK)', 'Home_Thr', 'Home_Launch%', 'Home_Launch%_1',
        'Home_AvgLen', 'Home_AvgLen_1', 'Home_Att_2', 'Home_Att_3', 'Home_Launch%_2', 'Home_Launch%_3', 'Home_AvgLen_2',
        'Home_AvgLen_3', 'Home_Opp', 'Home_Stp', 'Home_Stp%', 'Home_#OPA', 'Home_#OPA/90', 'Home_Gls', 'Home_Sh',
        'Home_SoT', 'Home_SoT%', 'Home_Sh/90', 'Home_SoT/90', 'Home_G/Sh', 'Home_G/SoT', 'Home_Dist', 'Home_npxG/Sh',
        'Home_G-xG', 'Home_np:G-xG', 'Home_TotDist', 'Home_PrgDist', 'Home_xA', 'Home_A-xAG', 'Home_KP', 'Home_1/3',
        'Home_PPA', 'Home_CrsPA', 'Home_Live', 'Home_Dead', 'Home_TB', 'Home_Sw', 'Home_Crs', 'Home_TI', 'Home_In',
        'Home_Out', 'Home_Str', 'Home_Off', 'Home_Blocks', 'Home_SCA', 'Home_SCA90', 'Home_PassLive', 'Home_PassLive_1',
        'Home_PassDead', 'Home_PassDead_1', 'Home_TO', 'Home_TO_1', 'Home_Fld', 'Home_Fld_1', 'Home_Def', 'Home_Def_1',
        'Home_GCA', 'Home_GCA90', 'Home_PassLive_2', 'Home_PassLive_3', 'Home_PassDead_2', 'Home_PassDead_3', 'Home_TO_2',
        'Home_TO_3', 'Home_Fld_2', 'Home_Fld_3', 'Home_Def_2', 'Home_Def_3', 'Home_Tkl', 'Home_Tkl_1', 'Home_TklW',
        'Home_Def 3rd', 'Home_Mid 3rd', 'Home_Att 3rd', 'Home_Tkl_2', 'Home_Tkl_3', 'Home_Tkl%', 'Home_Lost',
        'Home_Pass', 'Home_Int', 'Home_Tkl+Int', 'Home_Clr', 'Home_Err', 'Home_Touches', 'Home_Def Pen', 'Home_Att Pen',
        'Home_Succ', 'Home_Succ%', 'Home_Tkld', 'Home_Tkld%', 'Home_Carries', 'Home_CPA', 'Home_Mis', 'Home_Dis',
        'Home_Rec', 'Home_PrgR', 'Home_Mn/MP', 'Home_Min%', 'Home_Mn/Start', 'Home_Subs', 'Home_Mn/Sub', 'Home_unSub',
        'Home_PPM', 'Home_onG', 'Home_onGA', 'Home_+/-', 'Home_+/-90', 'Home_onxG', 'Home_onxGA', 'Home_xG+/-',
        'Home_xG+/-90', 'Home_2CrdY', 'Home_Fls', 'Home_PKwon', 'Home_PKcon', 'Home_Recov', 'Home_Won', 'Home_Won%', 'xG',
        'Away_Rk', 'Away_MP', 'Away_W', 'Away_D', 'Away_L', 'Away_GF', 'Away_GA', 'Away_GD', 'Away_Pts', 'Away_Pts/MP',
        'Away_xG', 'Away_xGA', 'Away_xGD', 'Away_xGD/90', 'Away_HMP', 'Away_HW', 'Away_HD', 'Away_HL', 'Away_HGF',
        'Away_HGA', 'Away_HGD', 'Away_HPts', 'Away_HPts/MP', 'Away_HxG', 'Away_HxGA', 'Away_HxGD', 'Away_AxGD/90',
        'Away_AxGD/90_1', 'Away_AMP', 'Away_AW', 'Away_AD', 'Away_AL', 'Away_AGF', 'Away_AGA', 'Away_AGD', 'Away_APts',
        'Away_APts/MP', 'Away_AxG', 'Away_AxGA', 'Away_AxGD', 'Away_AxGD/90_2', 'Away_AxGD/90_3', 'Away_Age',
        'Away_Poss', 'Away_Ast', 'Away_Ast_1', 'Away_G+A', 'Away_G+A_1', 'Away_G-PK', 'Away_G-PK_1', 'Away_PK',
        'Away_PKatt', 'Away_CrdY', 'Away_CrdR', 'Away_npxG', 'Away_npxG_1', 'Away_xAG', 'Away_xAG_1', 'Away_npxG+xAG',
        'Away_npxG+xAG_1', 'Away_PrgC', 'Away_PrgP', 'Away_Ast_2', 'Away_Ast_3', 'Away_G+A_2', 'Away_G+A_3',
        'Away_G-PK_2', 'Away_G-PK_3', 'Away_G+A-PK', 'Away_xAG_2', 'Away_xAG_3', 'Away_xG+xAG', 'Away_npxG_2',
        'Away_npxG_3', 'Away_npxG+xAG_2', 'Away_npxG+xAG_3', 'Away_GA90', 'Away_SoTA', 'Away_Saves', 'Away_Save%',
        'Away_CS', 'Away_CS%', 'Away_PKA', 'Away_PKsv', 'Away_PKm', 'Away_Save%_1', 'Away_FK', 'Away_CK', 'Away_OG',
        'Away_PSxG', 'Away_PSxG/SoT', 'Away_PSxG+/-', 'Away_/90', 'Away_Cmp', 'Away_Att', 'Away_Att_1', 'Away_Cmp%',
        'Away_Att (GK)', 'Away_Thr', 'Away_Launch%', 'Away_Launch%_1', 'Away_AvgLen', 'Away_AvgLen_1', 'Away_Att_2',
        'Away_Att_3', 'Away_Launch%_2', 'Away_Launch%_3', 'Away_AvgLen_2', 'Away_AvgLen_3', 'Away_Opp', 'Away_Stp',
        'Away_Stp%', 'Away_#OPA', 'Away_#OPA/90', 'Away_Gls', 'Away_Sh', 'Away_SoT', 'Away_SoT%', 'Away_Sh/90',
        'Away_SoT/90', 'Away_G/Sh', 'Away_G/SoT', 'Away_Dist', 'Away_npxG/Sh', 'Away_G-xG', 'Away_np:G-xG',
        'Away_TotDist', 'Away_PrgDist', 'Away_xA', 'Away_A-xAG', 'Away_KP', 'Away_1/3', 'Away_PPA', 'Away_CrsPA',
        'Away_Live', 'Away_Dead', 'Away_TB', 'Away_Sw', 'Away_Crs', 'Away_TI', 'Away_In', 'Away_Out', 'Away_Str',
        'Away_Off', 'Away_Blocks', 'Away_SCA', 'Away_SCA90', 'Away_PassLive', 'Away_PassLive_1', 'Away_PassDead',
        'Away_PassDead_1', 'Away_TO', 'Away_TO_1', 'Away_Fld', 'Away_Fld_1', 'Away_Def', 'Away_Def_1', 'Away_GCA',
        'Away_GCA90', 'Away_PassLive_2', 'Away_PassLive_3', 'Away_PassDead_2', 'Away_PassDead_3', 'Away_TO_2',
        'Away_TO_3', 'Away_Fld_2', 'Away_Fld_3', 'Away_Def_2', 'Away_Def_3', 'Away_Tkl', 'Away_Tkl_1', 'Away_TklW',
        'Away_Def 3rd', 'Away_Mid 3rd', 'Away_Att 3rd', 'Away_Tkl_2', 'Away_Tkl_3', 'Away_Tkl%', 'Away_Lost',
        'Away_Pass', 'Away_Int', 'Away_Tkl+Int', 'Away_Clr', 'Away_Err', 'Away_Touches', 'Away_Def Pen', 'Away_Att Pen',
        'Away_Succ', 'Away_Succ%', 'Away_Tkld', 'Away_Tkld%', 'Away_Carries', 'Away_CPA', 'Away_Mis', 'Away_Dis',
        'Away_Rec', 'Away_PrgR', 'Away_Mn/MP', 'Away_Min%', 'Away_Mn/Start', 'Away_Subs', 'Away_Mn/Sub', 'Away_unSub',
        'Away_PPM', 'Away_onG', 'Away_onGA', 'Away_+/-', 'Away_+/-90', 'Away_onxG', 'Away_onxGA', 'Away_xG+/-',
        'Away_xG+/-90', 'Away_2CrdY', 'Away_Fls', 'Away_PKwon', 'Away_PKcon', 'Away_Recov', 'Away_Won', 'Away_Won%', 'xG.1'
    ]]



        
    target = match_results['Result']


    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled, target, label_encoder_home, label_encoder_away, scaler, home_stats, away_stats

def build_and_train_model(X_train, y_train):
    """Build and train the deep learning model."""
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=regularizers.l2(0.1)))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.1)))
    model.add(Dropout(0.15))
    model.add(Dense(3, activation='softmax'))  # Output layer with 3 units for 3 possible outcomes

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

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
    features_scaled, target, label_encoder_home, label_encoder_away, scaler, home_stats, away_stats = prepare_data('final.csv', decay_factor=0.1)

    average_accuracy = cross_validate_model(features_scaled, target, k=5)
    print(f'Average Accuracy from Cross-Validation: {average_accuracy:.2f}')
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

    # Build and train the model
    model = build_and_train_model(X_train, y_train)

    # Example input for prediction
    home_team = 'Crystal Palace'
    away_team = 'Liverpool'
    wk = '7'

    # Predict match result
    outcome_labels, predicted_probabilities = predict_match_result(model, scaler, label_encoder_home, label_encoder_away, home_stats, away_stats, home_team, away_team)
    
    print(f"Predicted probabilities for {home_team} vs. {away_team}:")
    for label, prob in zip(outcome_labels, predicted_probabilities):
        print(f"{label}: {prob:.2f}")