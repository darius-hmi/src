import os

def rename_csv_files(directory, old_prefix, new_prefix, start_year, end_year):
    """Rename CSV files in the specified directory."""
    # List all files in the directory
    for filename in os.listdir(directory):
        if filename.startswith(old_prefix) and filename.endswith('.csv'):
            try:
                # Extract the number from the old filename
                number = filename[len(old_prefix):-len('.csv')].strip('() ')
                
                # Determine the new filename
                if number.isdigit():
                    number = int(number)
                    year = start_year - number
                    new_filename = f"{new_prefix}{year}.csv"
                    
                    # Construct full old and new file paths
                    old_file_path = os.path.join(directory, filename)
                    new_file_path = os.path.join(directory, new_filename)
                    
                    # Rename the file
                    os.rename(old_file_path, new_file_path)
                    print(f"Renamed '{filename}' to '{new_filename}'")
                else:
                    print(f"Skipping file '{filename}' as it does not match expected format.")
            except Exception as e:
                print(f"Error renaming file '{filename}': {e}")

# Directory containing your CSV files
directory = 'data/tableStandings'

# Old and new prefixes
old_prefix = 'export ('
new_prefix = 'table'

# Range of years
start_year = 2023
end_year = 2010

rename_csv_files(directory, old_prefix, new_prefix, start_year, end_year)
