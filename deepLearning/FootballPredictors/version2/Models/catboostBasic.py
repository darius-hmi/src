import pandas as pd
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import sys
import os
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
import ipywidgets

def apply_scoreToResult_012(df):
    ftr_map = {'H': 1, 'D': 2, 'A': 0}
    df['Result'] = df['Result'].map(ftr_map)
    return df

def prepare_for_training(df):
    df = df[df['status'] == 'complete']
    df = df.drop(columns=
                   ["date_GMT", "status", "home_team_goal_count", "away_team_goal_count", "home_team_goal_count_half_time", "away_team_goal_count_half_time",
                    "odds_ft_home_team_win", "odds_ft_draw", "odds_ft_away_team_win", "home_team_season", "away_team_season", "total_goal_count", "total_goals_at_half_time"
    ])
    X = df.drop(columns=["Result"])
    y = df['Result']
    return df, X, y

def get_clean_prediction_data(df, week, season):
    df = df[df['Game Week'] == week]
    df = df[df['home_team_season'] == season]
    df = df.drop(columns=
                   ["date_GMT", "status", "home_team_goal_count", "away_team_goal_count", "home_team_goal_count_half_time", "away_team_goal_count_half_time",
                    "odds_ft_home_team_win", "odds_ft_draw", "odds_ft_away_team_win", "home_team_season", "away_team_season", "Result", "total_goal_count", "total_goals_at_half_time"
    ])
    return df
    
def fill_missing_data_prediction(trainData, predictionData):
    team_columns_home = [
        'home_team_corner_count', 'home_team_yellow_cards', 'home_team_red_cards', 'home_team_shots_x',
        'home_team_shots_on_target_x', 'home_team_shots_off_target_x', 'home_team_fouls_x', 'home_team_possession',
        'home_team_first_half_cards', 'home_team_second_half_cards', 'home_team_xg'
    ]
    
    team_columns_away = [
        'away_team_corner_count', 'away_team_yellow_cards', 'away_team_red_cards', 'away_team_shots_x',
        'away_team_shots_on_target_x', 'away_team_shots_off_target_x', 'away_team_fouls_x', 'away_team_possession',
        'away_team_first_half_cards', 'away_team_second_half_cards', 'away_team_xg'
    ]
    
    for index, row in predictionData.iterrows():
        home_team = row['home_team_name']
        away_team = row['away_team_name']
        
        for col in team_columns_home:
            home_team_data = trainData[trainData['home_team_name'] == home_team][col]
            
            if len(home_team_data) > 0:
                predictionData.at[index, col] = home_team_data.mean()
        
        for col in team_columns_away:
            away_team_data = trainData[trainData['away_team_name'] == away_team][col]
            
            if len(away_team_data) > 0:
                predictionData.at[index, col] = away_team_data.mean()

        # if pd.isna(row['total_goal_count']):
        #     home_goals = df[df['home_team_name'] == home_team]['total_goal_count']
        #     away_goals = df[df['away_team_name'] == away_team]['total_goal_count']
        #     if len(home_goals) > 0 and len(away_goals) > 0:
        #         df.at[index, 'total_goal_count'] = np.mean([home_goals.mean(), away_goals.mean()])
        #     elif len(home_goals) > 0:
        #         df.at[index, 'total_goal_count'] = home_goals.mean()
        #     elif len(away_goals) > 0:
        #         df.at[index, 'total_goal_count'] = away_goals.mean()
    return predictionData


data = pd.read_csv('../Data/finalPlus.csv')

dataAfterCleaningBeforeTrain = apply_scoreToResult_012(data)

readyForTrainData, X, y = prepare_for_training(dataAfterCleaningBeforeTrain)

X['referee'] = X['referee'].fillna("missing")

categorical_features = ['home_team_name', 'away_team_name', 'referee']
categorical_feature_indices = [X.columns.get_loc(col) for col in categorical_features]

cat_features = categorical_feature_indices

data2 = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = data2

train_pool = Pool(
    data = X_train,
    label=y_train,
    cat_features=cat_features
)

test_pool = Pool(
    data = X_test,
    label=y_test,
    cat_features=cat_features
)

model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.01,
    depth=7,
    l2_leaf_reg=10,
    loss_function='MultiClass',
    verbose=3,
    early_stopping_rounds=20
)

# Fit CatBoostClassifier
model.fit(train_pool, eval_set=test_pool, verbose=3, plot=True)



y_train_pred = model.predict(train_pool)
y_pred = model.predict(test_pool)

train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"CatBoost Model - Training Accuracy: {train_accuracy}")
print("Training Classification Report:")
print(classification_report(y_train, y_train_pred))

test_accuracy = accuracy_score(y_test, y_pred)
print(f"CatBoost Model - Test Accuracy: {test_accuracy}")
print("Test Classification Report:")
print(classification_report(y_test, y_pred))

# # Make predictions for new data
# pred = get_clean_prediction_data(dataAfterCleaningBeforeTrain, week=16, season='2024/2025')

# finalpred = fill_missing_data_prediction(X, pred)

# match_data = finalpred.copy()
# match_data_scaled = scaler.transform(match_data)

# predictions = model.predict_proba(match_data_scaled)

# finalpred['home_team_name'] = label_encoder.inverse_transform(finalpred['home_team_name'])
# finalpred['away_team_name'] = label_encoder.inverse_transform(finalpred['away_team_name'])

# for i, prediction in enumerate(predictions):
#     home_team = finalpred['home_team_name'].iloc[i]
#     away_team = finalpred['away_team_name'].iloc[i]

#     home_win_prob = prediction[1]
#     draw_prob = prediction[2]
#     away_win_prob = prediction[0]

#     print(f"{home_team} vs {away_team} - H: {home_win_prob * 100:.1f}%, D: {draw_prob * 100:.1f}%, A: {away_win_prob * 100:.1f}%")

