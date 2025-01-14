import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# Read in your data
df = pd.read_csv('final.csv')

# Function to calculate overall team strength, including goals scored, goals conceded, and win percentages
def calculate_strength(df):
    team_strength = {}

    teams = set(df['home_team_name']).union(set(df['away_team_name']))

    for team in teams:
        # Home games for the team
        home_matches = df[df['home_team_name'] == team]
        home_goals_scored = home_matches['home_team_goal_count'].sum()
        home_goals_conceded = home_matches['away_team_goal_count'].sum()
        home_wins = home_matches[home_matches['home_team_goal_count'] > home_matches['away_team_goal_count']].shape[0]
        home_losses = home_matches[home_matches['home_team_goal_count'] < home_matches['away_team_goal_count']].shape[0]
        
        # Away games for the team
        away_matches = df[df['away_team_name'] == team]
        away_goals_scored = away_matches['away_team_goal_count'].sum()
        away_goals_conceded = away_matches['home_team_goal_count'].sum()
        away_wins = away_matches[away_matches['away_team_goal_count'] > away_matches['home_team_goal_count']].shape[0]
        away_losses = away_matches[away_matches['away_team_goal_count'] < away_matches['home_team_goal_count']].shape[0]

        # Total games (home + away)
        total_goals_scored = home_goals_scored + away_goals_scored
        total_goals_conceded = home_goals_conceded + away_goals_conceded
        total_matches = home_matches.shape[0] + away_matches.shape[0]

        # Calculate average goals scored and conceded (both home and away)
        avg_goals_scored = total_goals_scored / total_matches if total_matches > 0 else 0
        avg_goals_conceded = total_goals_conceded / total_matches if total_matches > 0 else 0

        # Calculate win percentages
        total_wins = home_wins + away_wins
        win_percentage = total_wins / total_matches if total_matches > 0 else 0
        home_win_percentage = home_wins / home_matches.shape[0] if home_matches.shape[0] > 0 else 0
        away_win_percentage = away_wins / away_matches.shape[0] if away_matches.shape[0] > 0 else 0

        # Store results for the team
        team_strength[team] = {
            'avg_goals_scored': avg_goals_scored,
            'avg_goals_conceded': avg_goals_conceded,
            'win_percentage': win_percentage,
            'home_win_percentage': home_win_percentage,
            'away_win_percentage': away_win_percentage
        }

    return team_strength

# Calculate the team strengths
team_strength = calculate_strength(df)

# Assuming we have the following prior from the team_strength dictionary
# Example values (use actual ones from your team_strength data):
villa_home_lambda = team_strength['Aston Villa']['home_win_percentage'] * team_strength['Aston Villa']['avg_goals_scored']  # Adjusted for home performance
villa_away_lambda = team_strength['Aston Villa']['away_win_percentage'] * team_strength['Aston Villa']['avg_goals_scored']  # Adjusted for away performance
villa_conceded_lambda = team_strength['Aston Villa']['avg_goals_conceded']  # Average goals conceded

city_home_lambda = team_strength['Manchester City']['home_win_percentage'] * team_strength['Manchester City']['avg_goals_scored']  # Adjusted for home performance
city_away_lambda = team_strength['Manchester City']['away_win_percentage'] * team_strength['Manchester City']['avg_goals_scored']  # Adjusted for away performance
city_conceded_lambda = team_strength['Manchester City']['avg_goals_conceded']  # Average goals conceded

# Define the range of goals (0 to 5 goals is reasonable for football)
max_goals = 5

# Number of simulations
num_simulations = 100

# Store results
simulated_scores = []

# Run multiple simulations
for _ in range(num_simulations):
    # Generate Poisson distribution probabilities for goals scored by both teams
    villa_goals_probs = poisson.pmf(np.arange(0, max_goals+1), villa_home_lambda) if np.random.random() < 0.5 else poisson.pmf(np.arange(0, max_goals+1), villa_away_lambda)
    city_goals_probs = poisson.pmf(np.arange(0, max_goals+1), city_home_lambda) if np.random.random() < 0.5 else poisson.pmf(np.arange(0, max_goals+1), city_away_lambda)

    # Normalize the probabilities so they sum to 1
    villa_goals_probs /= villa_goals_probs.sum()  # Normalize Aston Villa's goal probabilities
    city_goals_probs /= city_goals_probs.sum()  # Normalize Manchester City's goal probabilities

    # Predict the score by generating a sample
    villa_predicted_score = np.random.choice(np.arange(0, max_goals+1), p=villa_goals_probs)
    city_predicted_score = np.random.choice(np.arange(0, max_goals+1), p=city_goals_probs)
    
    # Adjust prediction with goals conceded (i.e., use goals conceded as a factor for the opponent's attack)
    # For example, Aston Villa might be more likely to concede goals based on their own defensive weaknesses
    villa_predicted_score = max(0, villa_predicted_score - city_conceded_lambda)  # Adjust for conceded goals
    city_predicted_score = max(0, city_predicted_score - villa_conceded_lambda)  # Adjust for conceded goals

    # Append the result of each simulation
    simulated_scores.append((villa_predicted_score, city_predicted_score))

# Convert the results into a numpy array for easier analysis
simulated_scores = np.array(simulated_scores)

# Example: Calculate the average predicted score
avg_villa_score = simulated_scores[:, 0].mean()
avg_city_score = simulated_scores[:, 1].mean()

print(f"Average predicted score after {num_simulations} simulations: Aston Villa {avg_villa_score:.2f} - Manchester City {avg_city_score:.2f}")

# Optional: Visualize the goal distribution
plt.figure(figsize=(10, 5))
plt.bar(np.arange(0, max_goals+1), villa_goals_probs, alpha=0.6, label="Aston Villa")
plt.bar(np.arange(0, max_goals+1), city_goals_probs, alpha=0.6, label="Manchester City")
plt.xlabel("Goals")
plt.ylabel("Probability")
plt.title("Predicted Goal Distribution for Aston Villa vs Manchester City")
plt.legend()
plt.show()
