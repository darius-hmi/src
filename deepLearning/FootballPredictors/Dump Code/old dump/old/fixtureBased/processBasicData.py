import pandas as pd
import os

# Directory where the CSV files are stored
csv_dir = 'data'  # Replace with the path where your CSV files are saved

# List of CSV file names
csv_files = [f'E0 ({i}).csv' for i in range(7)]

# Read and concatenate all CSV files
df_list = [pd.read_csv(os.path.join(csv_dir, file)) for file in csv_files]
concatenated_df = pd.concat(df_list, ignore_index=True)

# Drop any columns that contain any null values
concatenated_df = concatenated_df.dropna(axis=1, how='any')

# Save the concatenated DataFrame to a new CSV file
concatenated_df.to_csv('final.csv', index=False)

print("All files have been concatenated, columns with null values have been removed, and the result saved to 'final.csv'")
