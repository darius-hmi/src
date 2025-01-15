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

data = pd.read_csv("2023/england-premier-league-matches-2023-to-2024-stats.csv")
data['Result'] = determine_result(data)
data = data.drop(columns=['timestamp', 'attendance', 'home_team_goal_timings', 'away_team_goal_timings', 'stadium_name', 'date_GMT', 'status', 'home_team_name', 'away_team_name', 'referee'])

# Compute the correlation matrix
correlation_matrix = data.corr(method="pearson")  # Use "spearman" for monotonic relationships

# Plot the correlation matrix as a heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
# plt.title("Correlation Matrix")
# plt.show()

# Compute correlations with the target (Result)
target_correlations = data.corr(method="spearman")['Result'].sort_values(ascending=False)

# Display the strongest correlations
print(target_correlations)
