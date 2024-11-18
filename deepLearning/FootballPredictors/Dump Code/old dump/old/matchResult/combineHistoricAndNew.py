import pandas as pd

# Load the two CSV files
df1 = pd.read_csv('filtered_historic.csv')
df2 = pd.read_csv('filtered_updated.csv')

# Combine them vertically (i.e., stack them)
df_combined = pd.concat([df1, df2], ignore_index=True)

# Save the combined DataFrame to a new CSV file
df_combined.to_csv('final.csv', index=False)