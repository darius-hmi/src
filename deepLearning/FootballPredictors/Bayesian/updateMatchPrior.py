import numpy as np
import pandas as pd
from scipy.stats import poisson

priors = pd.read_csv('priorFor2024.csv')
df = pd.read_csv('2024/england-premier-league-matches-2024-to-2025-stats.csv')
matches = df[df['status'] == 'complete']

matches['alpha_home'] = 0.0
matches['beta_home'] = 0.0
matches['alpha_away'] = 0.0
matches['beta_away'] = 0.0

dynamic_stats = priors.set_index('team').copy()

default_alpha_home = priors['average_prior_goals_scored_home'].mean()**2 / priors['var_prior_goals_scored_home'].mean()
default_beta_home = priors['average_prior_goals_scored_home'].mean() / priors['var_prior_goals_scored_home'].mean()
default_alpha_away = priors['average_prior_goals_scored_away'].mean()**2 / priors['var_prior_goals_scored_away'].mean()
default_beta_away = priors['average_prior_goals_scored_away'].mean() / priors['var_prior_goals_scored_away'].mean()


for index, match in matches.iterrows():
    home_team = match['home_team_name']
    away_team = match['away_team_name']
    
    if home_team not in dynamic_stats.index:
        dynamic_stats.loc[home_team] = {
            'alpha_home': default_alpha_home,
            'beta_home': default_beta_home,
            'alpha_away': default_alpha_away,
            'beta_away': default_beta_away,
        }
    
    # Check if away team exists in dynamic_stats, if not initialize
    if away_team not in dynamic_stats.index:
        dynamic_stats.loc[away_team] = {
            'alpha_home': default_alpha_home,
            'beta_home': default_beta_home,
            'alpha_away': default_alpha_away,
            'beta_away': default_beta_away,
        }

    home_alpha = dynamic_stats.loc[home_team, 'alpha_home']
    home_beta = dynamic_stats.loc[home_team, 'beta_home']
    away_alpha = dynamic_stats.loc[away_team, 'alpha_away']
    away_beta = dynamic_stats.loc[away_team, 'beta_away']
    
    matches.at[index, 'alpha_home'] = home_alpha
    matches.at[index, 'beta_home'] = home_beta
    matches.at[index, 'alpha_away'] = away_alpha
    matches.at[index, 'beta_away'] = away_beta
    
    dynamic_stats.loc[home_team, 'alpha_home'] += match['home_team_goal_count']
    dynamic_stats.loc[home_team, 'beta_home'] += 1
    dynamic_stats.loc[away_team, 'alpha_away'] += match['away_team_goal_count']
    dynamic_stats.loc[away_team, 'beta_away'] += 1

print(matches)

def predict_outcome(alpha_H, beta_H, alpha_A, beta_A, max_goals=10):
    lambda_H = alpha_H / beta_H
    lambda_A = alpha_A / beta_A
    
    home_goals_probs = [poisson.pmf(k, lambda_H) for k in range(max_goals)]
    away_goals_probs = [poisson.pmf(k, lambda_A) for k in range(max_goals)]
    
    home_win = sum(
        home_goals_probs[i] * sum(away_goals_probs[j] for j in range(i))
        for i in range(max_goals)
    )
    draw = sum(
        home_goals_probs[i] * away_goals_probs[i] for i in range(max_goals)
    )
    away_win = sum(
        away_goals_probs[i] * sum(home_goals_probs[j] for j in range(i))
        for i in range(max_goals)
    )
    
    if home_win > draw and home_win > away_win:
        return "H" 
    elif draw > home_win and draw > away_win:
        return "D"
    else:
        return "A" 

matches['predicted_result'] = ""
matches['actual_result'] = ""

for index, match in matches.iterrows():
    predict_result = predict_outcome(match['alpha_home'], match['beta_home'], match['alpha_away'], match['beta_away'])
    if match['home_team_goal_count'] > match['away_team_goal_count']:
        result = "H" 
    elif match['home_team_goal_count'] < match['away_team_goal_count']:
        result = "A" 
    else:
        result = "D" 
    matches.at[index, 'predicted_result'] = predict_result
    matches.at[index, 'actual_result'] = result
    

matches.to_csv('test.csv')


accuracy = (matches['actual_result'] == matches['predicted_result']).mean() * 100

print(f"Accuracy: {accuracy:.2f}%")

matches['random_result'] = np.random.choice(['H', 'A', 'D'], size=len(matches))
random_accuracy = (matches['random_result'] == matches['actual_result']).mean() * 100
print(f"Random prediction accuracy: {random_accuracy:.2f}%")


# matches['homeAlpha'] = 0.0
# matches['homeBeta'] = 0.0
# matches['awayAlpha'] = 0.0
# matches['awayBeta'] = 0.0

# team_dic = []

# for index, row in priors.iterrows():
#     team_dic.append({
#         'team': row['team'],
#         'alpha_home': row['average_prior_goals_scored_home']**2 / row['var_prior_goals_scored_home']**2,
#         'beta_home': row['average_prior_goals_scored_home'] / row['var_prior_goals_scored_home']**2,
#         'alpha_away': row['average_prior_goals_scored_away'] / row['var_prior_goals_scored_away']**2,
#         'beta_away': row['average_prior_goals_scored_away'] / row['var_prior_goals_scored_away']**2
#     })


# for index, row in matches.iterrows():
#     homeTeam = row['home_team_name']
#     awayTeam = row['away_team_name']
#     home_team_goals = row['home_team_goal_count']
#     away_team_goals = row['away_team_goal_count']