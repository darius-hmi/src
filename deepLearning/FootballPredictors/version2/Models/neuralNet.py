import pandas as pd
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import sys
import os
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import numpy as np

class CustomLabelEncoder:
    def __init__(self, min_value=1000, max_value=1000000):
        self.label_encoder = LabelEncoder()
        self.min_value = min_value
        self.max_value = max_value
        self.label_to_random_map = {}
        self.random_to_label_map = {}
        self.classes_ = None

    def fit(self, y):
        self.label_encoder.fit(y)
        self.classes_ = self.label_encoder.classes_

        # Generate unique random integers for each class
        random_values = np.random.choice(
            range(self.min_value, self.max_value),
            size=len(self.classes_),
            replace=False
        )
        self.label_to_random_map = dict(zip(self.classes_, random_values))
        self.random_to_label_map = {v: k for k, v in self.label_to_random_map.items()}
        return self

    def transform(self, y):
        return np.array([self.label_to_random_map[label] for label in y])

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y):
        return np.array([self.random_to_label_map[random_val] for random_val in y])

    def get_classes(self):
        return self.classes_

def apply_scoreToResult_012(df):
    ftr_map = {'H': 1, 'D': 0, 'A': 2}
    df['Result'] = df['Result'].map(ftr_map)
    return df

def apply_label_encoder(df):
    label_encoder = CustomLabelEncoder(min_value=1000, max_value=1000000)
    label_encoder_ref = CustomLabelEncoder(min_value=1000, max_value=1000000)
    df['referee'] = label_encoder_ref.fit_transform(df['referee'])
    all_teams = pd.concat([df['home_team_name'], df['away_team_name']]).unique()
    label_encoder.fit(all_teams)
    df['home_team_name'] = label_encoder.transform(df['home_team_name'])
    df['away_team_name'] = label_encoder.transform(df['away_team_name'])
    return df, label_encoder

def extract_start_year(season_str):
    return int(season_str.split('/')[0])

def exponential_decay_weight(df, decay_factor):
    df['Season_Year'] = df['home_team_season'].apply(extract_start_year)
    max_season = df['Season_Year'].max()  # Determine the most recent season
    weights = np.exp(-decay_factor * (max_season - df['Season_Year']))
    return weights

def calculate_weighted_averages(df, group_by_col, columns, weight_col='Weight'):
    return df.groupby(group_by_col).apply(
        lambda x: pd.Series({
            col: np.average(x[col], weights=x[weight_col]) if col in x.columns else np.nan
            for col in columns
        })
    ).reset_index()


def order_features_and_prepare_target(df):
    df = apply_scoreToResult_012(df)
    df, label_encoder = apply_label_encoder(df)
    
    return df, label_encoder

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

dataAfterCleaningBeforeTrain, label_encoder = order_features_and_prepare_target(data)

readyForTrainData, X, y = prepare_for_training(dataAfterCleaningBeforeTrain)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

early_stopping = EarlyStopping(monitor='val_loss',
                               patience=10,         
                               restore_best_weights=True, 
                               verbose=1) 

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    factor=0.5,
    patience=3,  
    min_lr=1e-6,
    verbose=1
)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),  # Input layer
    tf.keras.layers.Dense(100, activation='relu', kernel_initializer="he_normal", kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(100, activation='relu', kernel_initializer="he_normal", kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(100, activation='relu', kernel_initializer="he_normal", kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(3, activation='softmax')
])

optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9,
                                    nesterov=True)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])



history = model.fit(X_train_scaled, y_train, 
                    epochs=150, 
                    validation_split=0.2, 
                    callbacks=[early_stopping, lr_scheduler])  

test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=1)

print(f"Test Accuracy: {test_accuracy:.2f}")
y_train_pred_probs = model.predict(X_train_scaled)
y_train_pred = tf.argmax(y_train_pred_probs, axis=1).numpy()

y_test_pred_probs = model.predict(X_test_scaled)
y_test_pred = tf.argmax(y_test_pred_probs, axis=1).numpy()

train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Neural Network Model - Training Accuracy: {train_accuracy:.2f}")
print("Training Classification Report:")
print(classification_report(y_train, y_train_pred))

test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Neural Network Model - Test Accuracy: {test_accuracy:.2f}")
print("Test Classification Report:")
print(classification_report(y_test, y_test_pred))


pred = get_clean_prediction_data(dataAfterCleaningBeforeTrain, week=18, season='2024/2025')

finalpred = fill_missing_data_prediction(X, pred)

match_data = finalpred.copy()

match_data_scaled = scaler.transform(match_data)
predictions  = model.predict(match_data_scaled)

finalpred['home_team_name'] = label_encoder.inverse_transform(finalpred['home_team_name'])
finalpred['away_team_name'] = label_encoder.inverse_transform(finalpred['away_team_name'])


for i, prediction in enumerate(predictions):
    home_team = finalpred['home_team_name'].values[i]
    away_team = finalpred['away_team_name'].values[i]

    home_win_prob = prediction[1]
    draw_prob = prediction[0]
    away_win_prob = prediction[2]

    print(f"{home_team} vs {away_team} - H: {home_win_prob * 100:.0f}%, D: {draw_prob * 100:.0f}%, A: {away_win_prob * 100:.0f}%")



