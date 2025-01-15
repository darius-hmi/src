import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import shap

def determine_result(df):
    return df.apply(
        lambda row: 'H' if row['home_team_goal_count'] > row['away_team_goal_count'] 
                    else 'A' if row['home_team_goal_count'] < row['away_team_goal_count'] 
                    else 'D',
        axis=1
    )

data = pd.read_csv("2023/england-premier-league-matches-2023-to-2024-stats.csv")
data['Result'] = determine_result(data)

X = data[["Pre-Match PPG (Home)", "Pre-Match PPG (Away)", "home_team_corner_count", "away_team_corner_count", 'home_team_shots', 'away_team_shots', 'home_team_possession', 'away_team_possession']]  # Example stats
y = data["Result"] 

model = RandomForestClassifier()
model.fit(X, y)

feature_importance = model.feature_importances_
print("Feature Importance:", feature_importance)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X)
