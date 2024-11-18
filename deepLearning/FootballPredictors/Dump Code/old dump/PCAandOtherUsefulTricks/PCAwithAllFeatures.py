import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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

def prepare_data(file_path, decay_factor=0.1, n_components=50):
    """Load and prepare the dataset with time decay and PCA."""
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

    features = match_results[['Wk', 'Home', 'Away', 'Home_Rk', 'Home_MP', 'Home_W',
                              'Home_D', 'Home_L', 'Home_GF', 'Home_GA', 'Home_GD', 'Home_Pts', 'Home_Pts/MP',
                              'Home_xG', 'Home_xGA', 'Home_xGD', 'Home_xGD/90', 'Home_# Pl', 'Home_90s',
                              'Home_Cmp', 'Home_Att', 'Home_Cmp%', 'Home_TotDist', 'Home_PrgDist',
                              'Home_Cmp.1', 'Home_Att.1', 'Home_Cmp%.1', 'Home_Cmp.2', 'Home_Att.2', 
                              'Home_Cmp%.2', 'Home_Cmp.3', 'Home_Att.3', 'Home_Cmp%.3', 'Home_Ast', 
                              'Home_xAG', 'Home_xA', 'Home_A-xAG', 'Home_KP', 'Home_01-Mar', 'Home_PPA',
                              'Home_CrsPA', 'Home_PrgP', 'Home_Live', 'Home_Dead', 'Home_FK', 'Home_TB',
                              'Home_Sw', 'Home_Crs', 'Home_TI', 'Home_CK', 'Home_In', 'Home_Out', 
                              'Home_Str', 'Home_Off', 'Home_Blocks', 'Home_SCA', 'Home_SCA90', 
                              'Home_PassLive', 'Home_PassDead', 'Home_TO', 'Home_Sh', 'Home_Fld',
                              'Home_Def', 'Home_GCA', 'Home_GCA90', 'Home_PassLive.1', 'Home_PassDead.1',
                              'Home_TO.1', 'Home_Sh.1', 'Home_Fld.1', 'Home_Def.1', 'Home_Tkl',
                              'Home_TklW', 'Home_Def 3rd', 'Home_Mid 3rd', 'Home_Att 3rd', 'Home_Tkl.1',
                              'Home_Tkl%', 'Home_Lost', 'Home_Pass', 'Home_Int', 'Home_Tkl+Int', 
                              'Home_Clr', 'Home_Err', 'Home_Poss', 'Home_Touches', 'Home_Def Pen',
                              'Home_Att Pen', 'Home_Succ', 'Home_Succ%', 'Home_Tkld', 'Home_Tkld%', 
                              'Home_Carries', 'Home_PrgC', 'Home_CPA', 'Home_Mis', 'Home_Dis',
                              'Home_Rec', 'Home_PrgR', 'Home_HMP', 'Home_HW', 'Home_HD', 'Home_HL',
                              'Home_HGF', 'Home_HGA', 'Home_HGD', 'Home_HPts', 'Home_HPts/MP',
                              'Home_HxG', 'Home_HxGA', 'Home_HxGD', 'Home_HxGD/90', 'Home_AMP',
                              'Home_AW', 'Home_AD', 'Home_AL', 'Home_AGF', 'Home_AGA', 'Home_AGD',
                              'Home_APts', 'Home_APts/MP', 'Home_AxG', 'Home_AxGA', 'Home_AxGD',
                              'Home_AxGD/90', 'Home_Age', 'Home_Min', 'Home_Mn/MP', 'Home_Min%', 
                              'Home_Starts', 'Home_Mn/Start', 'Home_Subs', 'Home_Mn/Sub', 
                              'Home_unSub', 'Home_PPM', 'Home_onG', 'Home_onGA', 'Home_+/-', 
                              'Home_+/-90', 'Home_onxG', 'Home_onxGA', 'Home_xG+/-', 'Home_xG+/-90',
                              'Home_CrdY', 'Home_CrdR', 'Home_2CrdY', 'Home_Fls', 'Home_PKwon',
                              'Home_PKcon', 'Home_OG', 'Home_Recov', 'Home_Won', 'Home_Won%', 
                              'Home_Gls', 'Home_G+A', 'Home_G-PK', 'Home_PK', 'Home_PKatt', 
                              'Home_npxG', 'Home_npxG+xAG', 'Home_Gls.1', 'Home_Ast.1', 
                              'Home_G+A.1', 'Home_G-PK.1', 'Home_G+A-PK', 'Home_xG.1', 
                              'Home_xAG.1', 'Home_xG+xAG', 'Home_npxG.1', 'Home_npxG+xAG.1', 
                              'Home_GA90', 'Home_SoTA', 'Home_Saves', 'Home_Save%', 'Home_CS',
                              'Home_CS%', 'Home_PKA', 'Home_PKsv', 'Home_PKm', 'Home_PSxG',
                              'Home_PSxG/SoT', 'Home_PSxG+/-', 'Home_/90', 'Home_Att (GK)', 
                              'Home_Thr', 'Home_Launch%', 'Home_AvgLen', 'Home_Launch%.1',
                              'Home_AvgLen.1', 'Home_Opp', 'Home_Stp', 'Home_Stp%', 'Home_#OPA', 
                              'Home_#OPA/90', 'Home_SoT', 'Home_SoT%', 'Home_Sh/90', 'Home_SoT/90', 
                              'Home_G/Sh', 'Home_G/SoT', 'Home_Dist', 'Home_npxG/Sh', 
                              'Home_G-xG', 'Home_np:G-xG', 'Home_Weekly_Wages_in_GBP', 'Home_Annual_Wages_in_GBP',
                              'xG', 'Away_Rk', 'Away_MP', 'Away_W', 
                              'Away_D', 'Away_L', 'Away_GF', 'Away_GA', 'Away_GD', 'Away_Pts',
                              'Away_Pts/MP', 'Away_xG', 'Away_xGA', 'Away_xGD', 'Away_xGD/90',
                              'Away_# Pl', 'Away_90s', 'Away_Cmp', 'Away_Att', 'Away_Cmp%',
                              'Away_TotDist', 'Away_PrgDist', 'Away_Cmp.1', 'Away_Att.1',
                              'Away_Cmp%.1', 'Away_Cmp.2', 'Away_Att.2', 'Away_Cmp%.2', 
                              'Away_Cmp.3', 'Away_Att.3', 'Away_Cmp%.3', 'Away_Ast', 
                              'Away_xAG', 'Away_xA', 'Away_A-xAG', 'Away_KP', 'Away_01-Mar',
                              'Away_PPA', 'Away_CrsPA', 'Away_PrgP', 'Away_Live', 'Away_Dead',
                              'Away_FK', 'Away_TB', 'Away_Sw', 'Away_Crs', 'Away_TI', 'Away_CK',
                              'Away_In', 'Away_Out', 'Away_Str', 'Away_Off', 'Away_Blocks',
                              'Away_SCA', 'Away_SCA90', 'Away_PassLive', 'Away_PassDead', 'Away_TO',
                              'Away_Sh', 'Away_Fld', 'Away_Def', 'Away_GCA', 'Away_GCA90',
                              'Away_PassLive.1', 'Away_PassDead.1', 'Away_TO.1', 'Away_Sh.1',
                              'Away_Fld.1', 'Away_Def.1', 'Away_Tkl', 'Away_TklW', 'Away_Def 3rd',
                              'Away_Mid 3rd', 'Away_Att 3rd', 'Away_Tkl.1', 'Away_Tkl%',
                              'Away_Lost', 'Away_Pass', 'Away_Int', 'Away_Tkl+Int', 'Away_Clr',
                              'Away_Err', 'Away_Poss', 'Away_Touches', 'Away_Def Pen', 
                              'Away_Att Pen', 'Away_Succ', 'Away_Succ%', 'Away_Tkld', 
                              'Away_Tkld%', 'Away_Carries', 'Away_PrgC', 'Away_CPA', 'Away_Mis',
                              'Away_Dis', 'Away_Rec', 'Away_PrgR', 'Away_HMP', 'Away_HW', 
                              'Away_HD', 'Away_HL', 'Away_HGF', 'Away_HGA', 'Away_HGD', 
                              'Away_HPts', 'Away_HPts/MP', 'Away_HxG', 'Away_HxGA', 
                              'Away_HxGD', 'Away_HxGD/90', 'Away_AMP', 'Away_AW', 'Away_AD',
                              'Away_AL', 'Away_AGF', 'Away_AGA', 'Away_AGD', 'Away_APts',
                              'Away_APts/MP', 'Away_AxG', 'Away_AxGA', 'Away_AxGD',
                              'Away_AxGD/90', 'Away_Age', 'Away_Min', 'Away_Mn/MP',
                              'Away_Min%', 'Away_Starts', 'Away_Mn/Start', 'Away_Subs',
                              'Away_Mn/Sub', 'Away_unSub', 'Away_PPM', 'Away_onG', 
                              'Away_onGA', 'Away_+/-', 'Away_+/-90', 'Away_onxG', 
                              'Away_onxGA', 'Away_xG+/-', 'Away_xG+/-90', 'Away_CrdY', 
                              'Away_CrdR', 'Away_2CrdY', 'Away_Fls', 'Away_PKwon',
                              'Away_PKcon', 'Away_OG', 'Away_Recov', 'Away_Won', 
                              'Away_Won%', 'Away_Gls', 'Away_G+A', 'Away_G-PK', 'Away_PK', 
                              'Away_PKatt', 'Away_npxG', 'Away_npxG+xAG', 'Away_Gls.1', 
                              'Away_Ast.1', 'Away_G+A.1', 'Away_G-PK.1', 'Away_G+A-PK',
                              'Away_xG.1', 'Away_xAG.1', 'Away_xG+xAG', 'Away_npxG.1', 
                              'Away_npxG+xAG.1', 'Away_GA90', 'Away_SoTA', 'Away_Saves',
                              'Away_Save%', 'Away_CS', 'Away_CS%', 'Away_PKA', 'Away_PKsv',
                              'Away_PKm', 'Away_PSxG', 'Away_PSxG/SoT', 'Away_PSxG+/-', 
                              'Away_/90', 'Away_Att (GK)', 'Away_Thr', 'Away_Launch%', 
                              'Away_AvgLen', 'Away_Launch%.1', 'Away_AvgLen.1', 
                              'Away_Opp', 'Away_Stp', 'Away_Stp%', 'Away_#OPA', 
                              'Away_#OPA/90', 'Away_SoT', 'Away_SoT%', 'Away_Sh/90',
                              'Away_SoT/90', 'Away_G/Sh', 'Away_G/SoT', 'Away_Dist',
                              'Away_npxG/Sh', 'Away_G-xG', 'Away_np:G-xG', 
                              'Away_Weekly_Wages_in_GBP', 'Away_Annual_Wages_in_GBP', 'xG.1']]

    target = match_results['Result']

    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Apply PCA
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features_scaled)

    return features_pca, target, label_encoder_home, label_encoder_away, scaler, pca, home_stats, away_stats

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

def predict_match_result(model, scaler, pca, label_encoder_home, label_encoder_away, home_stats, away_stats, home_team, away_team):
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

    # Normalize and apply PCA to example match data
    example_match_scaled = scaler.transform(example_match)
    example_match_pca = pca.transform(example_match_scaled)

    # Predict
    predictions = model.predict(example_match_pca)
    predicted_result = np.argmax(predictions[0])
    predicted_probability = predictions[0][predicted_result]  # Probability of the predicted outcome

    # Map result to outcome
    result_map = {0: 'Draw', 1: 'Home Win', 2: 'Away Win'}
    predicted_outcome = result_map[predicted_result]

    return predicted_outcome, predicted_probability

# Main workflow
if __name__ == "__main__":
    # Load and prepare data with decay factor and PCA
    features_pca, target, label_encoder_home, label_encoder_away, scaler, pca, home_stats, away_stats = prepare_data('data.csv', decay_factor=0.1, n_components=50)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_pca, target, test_size=0.2, random_state=42)

    # Build and train the model
    model = build_and_train_model(X_train, y_train)

    # Example input for prediction
    home_team = 'Bournemouth'
    away_team = 'Chelsea'
    wk = '4'

    # Predict match result
    predicted_outcome, predicted_probability = predict_match_result(model, scaler, pca, label_encoder_home, label_encoder_away, home_stats, away_stats, home_team, away_team)
    
    print(f"Predicted result for {home_team} vs. {away_team}: {predicted_outcome} with probability {predicted_probability:.2f}")
