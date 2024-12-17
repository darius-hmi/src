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

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'nn_model.joblib')
scaler_path = os.path.join(current_dir, 'scaler.joblib')
processed_data_path = os.path.join(current_dir, 'processed_data.csv')
label_encoder_path = os.path.join(current_dir, 'label_encoder.joblib')

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
    all_teams = pd.concat([df['Home'], df['Away']]).unique()
    label_encoder.fit(all_teams)
    df['Home'] = label_encoder.transform(df['Home'])
    df['Away'] = label_encoder.transform(df['Away'])
    return df, label_encoder

def extract_start_year(season_str):
    """Extract the starting year from the season string in 'YYYY/YYYY' format."""
    return int(season_str.split('/')[0])

def exponential_decay_weight(df, decay_factor):
    """Apply exponential decay to the season weights."""

    df['Season_Year'] = df['Season'].apply(extract_start_year)
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
    df = df.rename(columns={'xG': 'Home_xG_Base', 'xG.1': 'Away_xG_Base'})
    home_prefix = 'Home_'
    away_prefix = 'Away_'
    home_columns = [col for col in df.columns if col.startswith(home_prefix)]
    away_columns = [col for col in df.columns if col.startswith(away_prefix)]
    df['Weight'] = exponential_decay_weight(df, decay_factor=0.25)
    X = df[[
        'Wk', 'Home', 'Away'] + home_columns + away_columns
    ]
    y = df['Result']
    df_JustModel = pd.concat([X, y], axis=1)
    return X, y, df, df_JustModel, label_encoder



def apply_form_and_last3_goals(df):
    team_form = {}
    df['Home_Form'] = 7.5  # Initial average form
    df['Away_Form'] = 7.5  # Initial average form
    # Iterate over each row (match) in the DataFrame
    for idx, row in df.iterrows():
        home_team = row['Home']
        away_team = row['Away']
        result = row['Result']  # Full-time result (H, A, D)

        # Initialize form for teams if they are not already in the dictionary
        if home_team not in team_form:
            team_form[home_team] = []
        if away_team not in team_form:
            team_form[away_team] = []

        # Calculate points for the current match
        home_points = get_points(result, is_home=True)
        away_points = get_points(result, is_home=False)

        # Add the current points to the team's form (rolling list of points)
        team_form[home_team].append(home_points)
        team_form[away_team].append(away_points)

        # If the team has played more than 5 games, calculate the form as the sum of the last 5 games
        if len(team_form[home_team]) > 5:
            df.at[idx, 'Home_Form'] = sum(team_form[home_team][-6:])
        if len(team_form[away_team]) > 5:
            df.at[idx, 'Away_Form'] = sum(team_form[away_team][-6:])



    df['Home_Form2'] = 7.5  # Initial average form
    df['Away_Form2'] = 7.5


    # Initialize separate dictionaries for home and away forms
    home_team_form = {}
    away_team_form = {}

    # Iterate over each row (match) in the DataFrame
    for idx, row in df.iterrows():
        home_team = row['Home']
        away_team = row['Away']
        result = row['Result']  # Full-time result (H, A, D)

        # Initialize form for teams if they are not already in the dictionary
        if home_team not in home_team_form:
            home_team_form[home_team] = []
        if away_team not in away_team_form:
            away_team_form[away_team] = []

        # Calculate points for the current match
        home_points = get_points(result, is_home=True)
        away_points = get_points(result, is_home=False)

        # Add the current points to the team's home and away form (rolling list of points)
        home_team_form[home_team].append(home_points)
        away_team_form[away_team].append(away_points)

        # If the team has played more than 5 home or away games, calculate the form as the sum of the last 5 games
        if len(home_team_form[home_team]) > 5:
            df.at[idx, 'Home_Form2'] = sum(home_team_form[home_team][-6:])
        if len(away_team_form[away_team]) > 5:
            df.at[idx, 'Away_Form2'] = sum(away_team_form[away_team][-6:])
    

    # Initialize a dictionary to keep track of the last 3 games' goals for each team
    team_goals = {}

    # Example of how you might load your data
    # Add new columns to store the total goals scored and conceded in the last 3 games
    df['Home_Goals_Last_3'] = 0
    df['Away_Goals_Last_3'] = 0
    df['Home_Goals_Conceded_Last_3'] = 0
    df['Away_Goals_Conceded_Last_3'] = 0

    # Iterate over each row in the dataframe (match data)
    for idx, row in df.iloc[:-1].iterrows():
        home_team = row['Home']
        away_team = row['Away']
        score = row['Score']  # Full-time score, e.g., '2-1'

        # Get the number of goals scored by both teams
        home_goals, away_goals = get_goals_from_score(score)

        # Initialize the goal list if it's the first match for the teams
        if home_team not in team_goals:
            team_goals[home_team] = {'scored': [], 'conceded': []}
        if away_team not in team_goals:
            team_goals[away_team] = {'scored': [], 'conceded': []}

        # Append the goals scored and conceded in the current match
        team_goals[home_team]['scored'].append(home_goals)
        team_goals[away_team]['scored'].append(away_goals)
        team_goals[home_team]['conceded'].append(away_goals)
        team_goals[away_team]['conceded'].append(home_goals)

        # For home team: calculate the total goals scored in the last 3 games excluding the current game
        if len(team_goals[home_team]['scored']) > 1:  # If there are at least 2 games, calculate
            df.at[idx, 'Home_Goals_Last_3'] = sum(team_goals[home_team]['scored'][-4:-1])  # sum last 3 excluding current game
        if len(team_goals[home_team]['conceded']) > 1:  # If there are at least 2 games, calculate
            df.at[idx, 'Home_Goals_Conceded_Last_3'] = sum(team_goals[home_team]['conceded'][-4:-1])  # sum last 3 excluding current game

        # For away team: calculate the total goals scored in the last 3 games excluding the current game
        if len(team_goals[away_team]['scored']) > 1:  # If there are at least 2 games, calculate
            df.at[idx, 'Away_Goals_Last_3'] = sum(team_goals[away_team]['scored'][-4:-1])  # sum last 3 excluding current game
        if len(team_goals[away_team]['conceded']) > 1:  # If there are at least 2 games, calculate
            df.at[idx, 'Away_Goals_Conceded_Last_3'] = sum(team_goals[away_team]['conceded'][-4:-1])  # sum last 3 excluding current game

    last_row_idx = df.index[-1]
    last_row = df.iloc[-1]
    home_team = last_row['Home']
    away_team = last_row['Away']

    # Compute values based on the last 3 matches
    if len(team_goals[home_team]['scored']) >= 3:
        df.at[last_row_idx, 'Home_Goals_Last_3'] = sum(team_goals[home_team]['scored'][-3:])
    if len(team_goals[home_team]['conceded']) >= 3:
        df.at[last_row_idx, 'Home_Goals_Conceded_Last_3'] = sum(team_goals[home_team]['conceded'][-3:])

    if len(team_goals[away_team]['scored']) >= 3:
        df.at[last_row_idx, 'Away_Goals_Last_3'] = sum(team_goals[away_team]['scored'][-3:])
    if len(team_goals[away_team]['conceded']) >= 3:
        df.at[last_row_idx, 'Away_Goals_Conceded_Last_3'] = sum(team_goals[away_team]['conceded'][-3:])
    
    return df

data = pd.read_csv('fianlForReal.csv')
#data = data.drop(columns=['Home_Form', 'Away_Form', 'Home_Form2', 'Away_Form2', 'Home_Goals_Last_3', 'Away_Goals_Last_3', 'Home_Goals_Conceded_Last_3', 'Away_Goals_Conceded_Last_3'])
X, y, df, df_JustModel, label_encoder = order_features_and_prepare_target(data)

df.to_csv(processed_data_path, index=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

early_stopping = EarlyStopping(monitor='val_loss',  # Monitor validation loss
                               patience=10,         # Stop after 10 epochs without improvement
                               restore_best_weights=True,  # Restore the best weights after stopping
                               verbose=1) 

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    factor=0.5,
    patience=3,  # Try lowering patience
    min_lr=1e-6,  # Lower the minimum learning rate
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

optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9,
                                    nesterov=True)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])



history = model.fit(X_train_scaled, y_train, 
                    epochs=100, 
                    validation_split=0.2, # Set the maximum number of epochs you want to run
                    callbacks=[early_stopping, lr_scheduler])  # Add EarlyStopping callback

test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=1)

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(label_encoder, label_encoder_path)


print(f"Test Accuracy: {test_accuracy:.2f}")


y_train_pred_probs = model.predict(X_train_scaled)
y_train_pred = tf.argmax(y_train_pred_probs, axis=1).numpy()

y_test_pred_probs = model.predict(X_test_scaled)
y_test_pred = tf.argmax(y_test_pred_probs, axis=1).numpy()

# Training data evaluation
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Neural Network Model - Training Accuracy: {train_accuracy:.2f}")
print("Training Classification Report:")
print(classification_report(y_train, y_train_pred))

# Test data evaluation
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Neural Network Model - Test Accuracy: {test_accuracy:.2f}")
print("Test Classification Report:")
print(classification_report(y_test, y_test_pred))

home_prefix = 'Home_'
away_prefix = 'Away_'
home_columns = [col for col in df.columns if col.startswith(home_prefix)]
away_columns = [col for col in df.columns if col.startswith(away_prefix)]

home_stats = calculate_weighted_averages(df, 'Home', home_columns)
away_stats = calculate_weighted_averages(df, 'Away', away_columns)

# Set the index for easy lookup
home_stats.set_index('Home', inplace=True)
away_stats.set_index('Away', inplace=True)

# Function to prepare match data
def prepare_match_data(data, home_team, away_team, label_encoder):
    home_team_encoded = label_encoder.transform([home_team])[0]
    away_team_encoded = label_encoder.transform([away_team])[0]

    home_stats_row = home_stats.loc[home_team_encoded]
    away_stats_row = away_stats.loc[away_team_encoded]

    example_match = pd.DataFrame({
        'Wk': [data['Wk'].max() + 1],
        'Home': [home_team_encoded],
        'Away': [away_team_encoded],
        **home_stats_row.to_dict(),
        **away_stats_row.to_dict()
    })

    return example_match

# List of matches to predict
matches_to_predict = [
    ('Arsenal', 'Everton'),
    ('Liverpool', 'Fulham'),
    ('Newcastle Utd', 'Leicester City'),
    ('Wolves', 'Crystal Palace'),
    ('Nott\'ham Forest', 'Aston Villa'),
    ('Brighton', 'Crystal Palace'),
    ('Manchester City', 'Manchester Utd'),
    ('Chelsea', 'Brentford'),
    ('Southampton', 'Tottenham'),
    ('Bournemouth', 'West Ham')
]

# Iterate over the matches
for home_team, away_team in matches_to_predict:
    match_data = prepare_match_data(data, home_team, away_team, label_encoder)

    # Normalize match data
    match_data_scaled = scaler.transform(match_data)

    # Predict probabilities
    probabilities = model.predict(match_data_scaled)[0]

    # Decode probabilities
    home_win_prob = probabilities[1]
    draw_prob = probabilities[0]
    away_win_prob = probabilities[2]

    # Print the prediction results for the match
    print(f"{home_team} vs {away_team} - H: {home_win_prob * 100:.0f}%, D: {draw_prob * 100:.0f}%, A: {away_win_prob * 100:.0f}%")


