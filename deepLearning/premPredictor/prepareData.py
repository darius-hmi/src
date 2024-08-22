import pandas as pd
import os

# Base directory where the season folders are located
base_directory = 'data'

# List of season folders
seasons = ['2023-2024', '2022-2023', '2021-2022', '2020-2021', '2019-2020']

# Initialize an empty list to hold each DataFrame
combined_tables = []

# Iterate through each season folder
for season in seasons:
    # Define the path to the CSV file for table_1 in the current season folder
    file_path = os.path.join(base_directory, season, 'table_1.csv')
    
    # Check if the file exists
    if os.path.exists(file_path):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        # Add a Season column with the current season
        df['Season'] = season
        
        # Append the DataFrame to the list
        combined_tables.append(df)
    else:
        print(f"File {file_path} does not exist.")

# Concatenate all DataFrames in the list into a single DataFrame
combined_df = pd.concat(combined_tables, ignore_index=True)

# Save the combined DataFrame to a new CSV file
output_file = os.path.join(base_directory, 'combined_table_1.csv')
combined_df.to_csv(output_file, index=False)

print(f"Combined table_1 saved to {output_file}")
