import pandas as pd
import glob
import os

# List all CSV files in the specified directory
file_pattern = 'data/tableStandings/table*.csv'
file_list = glob.glob(file_pattern)

# Initialize an empty list to hold DataFrames
dfs = []

for file in file_list:
    # Normalize the file path
    file = os.path.normpath(file)
    
    print(f"Processing file: {file}")  # Debugging line

    # Extract the filename from the full path
    base_name = os.path.basename(file)
    
    if 'table' in base_name and '.csv' in base_name:
        try:
            # Extract the year from the filename
            year = int(base_name.split('table')[1].split('.csv')[0])
        except ValueError:
            print(f"Skipping file {file} due to unexpected filename format.")
            continue
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file)
        
        # Add a 'year' column
        df['year'] = year
        
        # Append the DataFrame to the list
        dfs.append(df)
    else:
        print(f"Skipping file {file} as it does not match the expected pattern.")

# Concatenate all DataFrames into a single DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

# Save the combined DataFrame to a new CSV file
combined_df.to_csv('combinedTables.csv', index=False)

print("Data combined and saved to 'combined_data.csv'")
