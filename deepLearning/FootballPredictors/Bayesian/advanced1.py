import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def determine_result(df):
    return df.apply(
        lambda row: 1 if row['home_team_goal_count'] > row['away_team_goal_count'] 
                    else -1 if row['home_team_goal_count'] < row['away_team_goal_count'] 
                    else 0,
        axis=1
    )


import pymc as pm
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import arviz as az

def main():
    # Load and preprocess data
    data = pd.read_csv("2023/england-premier-league-matches-2023-to-2024-stats.csv")
    data['Result'] = determine_result(data)
    # Define features and target
    features = ['home_ppg', 'away_ppg', 'home_team_shots', 'away_team_shots',
                'home_team_shots_on_target', 'away_team_shots_on_target',
                'home_team_possession', 'away_team_possession']
    target = 'Result'
    
    X = data[features]
    y = data[target]
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # PyMC model
    with pm.Model() as model:
        # Priors for weights and intercept
        intercept = pm.Normal("intercept", mu=0, sigma=1)
        weights = pm.Normal("weights", mu=0, sigma=1, shape=X_train.shape[1])
        
        # Linear model
        logits = intercept + pm.math.dot(X_train, weights)
        
        # Likelihood (sigmoid activation)
        p = pm.math.sigmoid(logits)
        observed = pm.Bernoulli("observed", p=p, observed=(y_train + 1) / 2)  # Convert -1/0/1 to 0/1
        
        # Sampling
        trace = pm.sample(100, tune=100, return_inferencedata=True)
        
        # Posterior predictive
        posterior_predictive = pm.sample_posterior_predictive(trace)
    
    # Print results
    # Visualize results with ArviZ
    # az.plot_posterior(trace)
    # az.plot_posterior_predictive(posterior_predictive, ref_val=y_train)
    # plt.show()
    print(posterior_predictive)

# Add safeguard
if __name__ == "__main__":
    main()

