import pandas as pd
import os

# Define the list of seasons
seasons = ['2019-2020', '2020-2021', '2021-2022', '2022-2023', '2023-2024']

# Define the base directory for the data
base_dir = 'data'

def determine_result(score):
    try:
        # Replace various types of dashes and special characters with a standard hyphen
        score = score.replace('–', '-').replace('—', '-').replace('−', '-')
        # Split the score into home and away scores
        home_score, away_score = map(int, score.split('-'))
        if home_score > away_score:
            return 'H'
        elif home_score < away_score:
            return 'A'
        else:
            return 'D'
    except Exception as e:
        # Handle any unexpected errors and provide debug information
        print(f"Error processing score '{score}': {e}")
        return 'Invalid'

for season in seasons:
    # Construct file paths
    team_stats_path = os.path.join(base_dir, season, 'updated_base_data.csv')
    fixtures_path = os.path.join(base_dir, season, 'fixtures', 'table_25.csv')
    temp_output_path = os.path.join(base_dir, season, 'fixtures_with_team_stats.csv')
    final_output_path = os.path.join(base_dir, season, 'fixtures_with_results.csv')
    
    # Load your data
    team_stats = pd.read_csv(team_stats_path)
    fixtures = pd.read_csv(fixtures_path)
    
    # Debug: Print column names to verify
    print(f"\nProcessing season {season}")
    print("Team Stats Columns:", team_stats.columns)
    print("Fixtures Columns:", fixtures.columns)
    
    # Ensure 'Squad' is present in team_stats
    if 'Squad' not in team_stats.columns:
        raise KeyError(f"'Squad' column not found in {team_stats_path}")
    
    # Ensure 'Home' and 'Away' columns are in fixtures
    if 'Home' not in fixtures.columns or 'Away' not in fixtures.columns:
        raise KeyError(f"'Home' and/or 'Away' column not found in {fixtures_path}")
    
    # Drop rows with NaN values in 'Home' or 'Away' columns
    fixtures = fixtures.dropna(subset=['Home', 'Away'])
    
    # Add prefixes to team_stats DataFrame
    home_stats = team_stats.add_prefix('Home_')
    away_stats = team_stats.add_prefix('Away_')
    
    # Merge team stats for Home Team
    fixtures = fixtures.merge(home_stats, how='left', left_on='Home', right_on='Home_Squad')
    
    # Merge team stats for Away Team
    fixtures = fixtures.merge(away_stats, how='left', left_on='Away', right_on='Away_Squad')
    
    # Optionally, drop the Home_Squad and Away_Squad if not needed
    fixtures = fixtures.drop(columns=['Home_Squad', 'Away_Squad'])
    
    # Apply the function to create the 'Result' column
    if 'Score' in fixtures.columns:
        fixtures['Result'] = fixtures['Score'].apply(determine_result)
    
        # Reorder columns to place 'Result' next to 'Score'
        score_index = fixtures.columns.get_loc('Score')
        fixtures.insert(score_index + 1, 'Result', fixtures.pop('Result'))
    
    # Save the combined data to a new CSV file
    fixtures.to_csv(final_output_path, index=False)
    
    print(f"Fixtures with team stats and results for season {season} have been successfully merged and saved to '{final_output_path}'")
