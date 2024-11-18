import pandas as pd

# Load your CSV file
df = pd.read_csv('data.csv')

# Specify the features you want to keep
features = ['Home', 'Result', 'Away', 'Wk', 'xG', 'Home_HW', 'Home_HPts/MP', 
            'Home_HPts', 'Home_HGD', 'Home_HL', 'Home_Gls', 'Home_+/-90', 'xG.1',
            'Away_APts/MP', 'Away_AGD', 'Away_AL', 'Away_APts', 'Away_GA90', 'Away_AW', 
            'Away_L', 'Away_Pts', 'Away_PPM', 'Away_Pts/MP', 'Season']

# Filter the DataFrame to keep only the specified features
df_filtered = df[features]

# Save the filtered DataFrame to a new CSV file
df_filtered.to_csv('filtered_historic.csv', index=False)
