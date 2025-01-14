import numpy as np
import pandas as pd
from scipy.stats import poisson

def get_average_and_variance(df):
    teams = set(df['home_team_name']).union(set(df['away_team_name']))
    results = []

    for team in teams:
        home_matches = df[df['home_team_name'] == team]
        average_prior_goals_scored_home = home_matches['home_team_goal_count'].mean()
        var_prior_goals_scored_home = home_matches['home_team_goal_count'].var()
        alpha_home = average_prior_goals_scored_home**2/var_prior_goals_scored_home
        beta_home = average_prior_goals_scored_home/var_prior_goals_scored_home

        away_matches = df[df['away_team_name'] == team]
        average_prior_goals_scored_away = away_matches['away_team_goal_count'].mean()
        var_prior_goals_scored_away = away_matches['away_team_goal_count'].var()
        alpha_away = average_prior_goals_scored_away**2/var_prior_goals_scored_away
        beta_away = average_prior_goals_scored_away/var_prior_goals_scored_away

        results.append({
            'team': team,
            'average_prior_goals_scored_home': average_prior_goals_scored_home,
            'var_prior_goals_scored_home': var_prior_goals_scored_home,
            'average_prior_goals_scored_away': average_prior_goals_scored_away,
            'var_prior_goals_scored_away': var_prior_goals_scored_away,
            'alpha_home': alpha_home,
            'beta_home': beta_home,
            'alpha_away' : alpha_away,
            'beta_away' : beta_away
        })

    new_df = pd.DataFrame(results)
    return new_df

df = pd.read_csv('2023/england-premier-league-matches-2023-to-2024-stats.csv')
new_df = get_average_and_variance(df)
print(new_df)
new_df.to_csv('priorFor2024.csv')

# Step 2: Define function to update priors with observed match data
def update_priors(alpha_H, beta_H, alpha_A, beta_A, goals_H, goals_A):
    updated_alpha_H = alpha_H + goals_H
    updated_beta_H = beta_H + 1
    updated_alpha_A = alpha_A + goals_A
    updated_beta_A = beta_A + 1
    
    return updated_alpha_H, updated_beta_H, updated_alpha_A, updated_beta_A

# Step 3: Define function to predict probabilities for future matches
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
    
    return {"Home Win": home_win, "Draw": draw, "Away Win": away_win}

# Step 4: Example usage of team-specific priors for predictions
team_1 = 'Arsenal'  # Replace with actual team name
team_2 = 'Tottenham Hotspur'  # Replace with actual team name

team_1_priors = new_df[new_df['team'] == team_1].iloc[0]
team_2_priors = new_df[new_df['team'] == team_2].iloc[0]

alpha_H_team_1 = team_1_priors['average_prior_goals_scored_home']**2 / team_1_priors['var_prior_goals_scored_home']
beta_H_team_1 = team_1_priors['average_prior_goals_scored_home'] / team_1_priors['var_prior_goals_scored_home']
alpha_A_team_2 = team_2_priors['average_prior_goals_scored_away']**2 / team_2_priors['var_prior_goals_scored_away']
beta_A_team_2 = team_2_priors['average_prior_goals_scored_away'] / team_2_priors['var_prior_goals_scored_away']

# Step 5: Update priors with observed match result
# Example observed match result: Team A vs Team B with Team A scoring 2 goals and Team B scoring 1 goal
# home_goals = 1
# away_goals = 2

# outcome_probs = predict_outcome(alpha_H_team_1, beta_H_team_1, alpha_A_team_2, beta_A_team_2)
# print("\nbefore updated prior Predictions:")
# print(f"  Home Win: {outcome_probs['Home Win']:.2%}")
# print(f"  Draw: {outcome_probs['Draw']:.2%}")
# print(f"  Away Win: {outcome_probs['Away Win']:.2%}")

# alpha_H_team_1, beta_H_team_1, alpha_A_team_2, beta_A_team_2 = update_priors(
#     alpha_H_team_1, beta_H_team_1, alpha_A_team_2, beta_A_team_2,
#     home_goals, away_goals
# )

# # Step 7: Predict probabilities for future matches with updated priors
# final_outcomes = predict_outcome(alpha_H_team_1, beta_H_team_1, alpha_A_team_2, beta_A_team_2)
# print("\nFinal Predictions for Future Match:")
# print(f"  Home Win: {final_outcomes['Home Win']:.2%}")
# print(f"  Draw: {final_outcomes['Draw']:.2%}")
# print(f"  Away Win: {final_outcomes['Away Win']:.2%}")





df2 = pd.read_csv('2024/england-premier-league-teams-2024-to-2025-stats.csv')

alpha_hat_home = alpha_H_team_1 + df2.loc[df2['common_name'] == team_1, 'goals_scored_home'].values[0]
beta_hat_home = beta_H_team_1 + df2.loc[df2['common_name'] == team_1, 'matches_played_home'].values[0]
alpha_hat_away = alpha_A_team_2 + df2.loc[df2['common_name'] == team_2, 'goals_scored_away'].values[0]
beta_hat_away = beta_A_team_2 + df2.loc[df2['common_name'] == team_2, 'matches_played_away'].values[0]

# Step 7: Predict probabilities for future matches with updated priors
final_outcomes = predict_outcome(alpha_hat_home, beta_hat_home, alpha_hat_away, beta_hat_away)
print("\n Predictions for Future Match:")
print(f"  Home Win: {final_outcomes['Home Win']:.2%}")
print(f"  Draw: {final_outcomes['Draw']:.2%}")
print(f"  Away Win: {final_outcomes['Away Win']:.2%}")




