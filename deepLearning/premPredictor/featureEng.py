import pandas as pd
import os

# Define the list of seasons
seasons = ['2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024']

# Define the base directory for the data
base_dir = 'data'

# List to hold all DataFrames
dfs = []

for season in seasons:
    # Construct file path for the fixtures with results CSV
    fixtures_with_results_path = os.path.join(base_dir, season, 'fixtures_with_results.csv')
    
    # Load the data
    df = pd.read_csv(fixtures_with_results_path, dtype=str)
    
    # Debug: Print column names and a few rows to verify
    print(f"\nProcessing season {season}")
    print("Sample of DataFrame:")
    print(df.head(10))
    
    # Ensure 'Home_Season' or 'Away_Season' is present
    if 'Home_Season' in df.columns:
        # Use the 'Home_Season' value directly as the 'Season' column
        df['Season'] = df['Home_Season']
    elif 'Away_Season' in df.columns:
        # Use the 'Away_Season' value directly as the 'Season' column
        df['Season'] = df['Away_Season']
    else:
        raise KeyError(f"'Home_Season' or 'Away_Season' column not found in {fixtures_with_results_path}")
    
    # Drop the columns 'Home_Season' and 'Away_Season' if not needed
    columns_to_drop = [
        'Home_Season', 'Away_Season', 'Attendance', 'Venue', 'Match Report', 'Notes', 'Home_Top Team Scorer', 'Home_Attendance',
          'Home_Top Team Scorer', 'Home_Goalkeeper', 'Home_Notes', 'Away_Top Team Scorer', 'Away_Attendance', 'Away_Goalkeeper', 'Away_Notes'
          ]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    # Append the DataFrame to the list
    dfs.append(df)

# Combine all DataFrames into one
combined_df = pd.concat(dfs, ignore_index=True)

# Save the combined DataFrame to a new CSV file
combined_output_path = os.path.join(base_dir, 'combined_fixtures_with_results.csv')
combined_df.to_csv(combined_output_path, index=False)

print(f"Combined fixtures with results have been successfully saved to '{combined_output_path}'")
