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
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

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

data = pd.read_csv('../Data/final.csv')

dataAfterCleaningBeforeTrain, label_encoder = order_features_and_prepare_target(data)

readyForTrainData, X, y = prepare_for_training(dataAfterCleaningBeforeTrain)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit a RandomForestClassifier to your data
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# Get feature importances
feature_importances = rf.feature_importances_
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get feature importances
feature_importances = rf.feature_importances_

# Sort the indices of features by their importance
sorted_indices = np.argsort(feature_importances)[::-1]  # Sort in descending order

# Get the top 50 important features
top_50_indices = sorted_indices[:50]
top_50_features = X.columns[top_50_indices]
top_50_importances = feature_importances[top_50_indices]

# Display and save the top 50 features
print("Top 50 Features and their Importances:")
for feature, importance in zip(top_50_features, top_50_importances):
    print(f"{feature}: {importance:.4f}")

# Save to a file
top_features_path = os.path.join(current_dir, 'top_50_features.txt')
with open(top_features_path, 'w') as f:
    for feature, importance in zip(top_50_features, top_50_importances):
        f.write(f"{feature}: {importance:.4f}\n")

print(f"Top 50 features saved to {top_features_path}")




# Get the indices of the 200 least important features
least_important_features_idx = np.argsort(feature_importances)[:300]

# Get the names of the 200 least important features (if available)
least_important_features = X.columns[least_important_features_idx]


# Save column names to a file
column_names_path = os.path.join(current_dir, 'column_names.txt')
with open(column_names_path, 'w') as f:
    for col in least_important_features:
        f.write(col + '\n')

print(f"Column names saved to {column_names_path}")

data2 = pd.read_csv('../Data/final.csv')

columns_to_remove = list(set(least_important_features))

# Ensure both "Home_" and "Away_" versions of the columns are removed if one exists
columns_to_remove_final = []
for col in columns_to_remove:
    if col.startswith('home_team_'):
        # Remove the corresponding Away column if it exists
        corresponding_away_col = 'away_team_' + col[10:]
        if corresponding_away_col not in columns_to_remove_final:
            columns_to_remove_final.append(corresponding_away_col)
    if col.startswith('away_team_'):
        # Remove the corresponding Home column if it exists
        corresponding_home_col = 'home_team_' + col[10:]
        if corresponding_home_col not in columns_to_remove_final:
            columns_to_remove_final.append(corresponding_home_col)
    if col not in columns_to_remove_final:
        columns_to_remove_final.append(col)

# Read the data
# Remove the columns listed in columns_to_remove_final
data4 = data2.drop(columns=columns_to_remove_final, errors='ignore')

# Apply the function to drop the columns
data4.to_csv('fianlForReal.csv')

# Print the least important features
print(f"Least important features: {least_important_features}")



