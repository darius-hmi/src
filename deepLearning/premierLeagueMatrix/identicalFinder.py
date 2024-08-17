import pandas as pd
import os

def load_csv_files(file_paths):
    """Load CSV files into a dictionary of DataFrames."""
    dataframes = {}
    for file_path in file_paths:
        if os.path.isfile(file_path):
            try:
                # Attempt to load CSV file
                df = pd.read_csv(file_path)
                dataframes[file_path] = df
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")
    return dataframes

def are_dataframes_identical(df1, df2):
    """Check if two DataFrames are identical."""
    return df1.equals(df2)

def check_csv_files_identical(file_paths):
    """Check if any CSV files in the list are identical."""
    all_dfs = load_csv_files(file_paths)
    
    if not all_dfs:
        print("No CSV files were loaded. Please check the file paths.")
        return
    
    # Compare each pair of DataFrames
    file_paths = list(all_dfs.keys())
    num_files = len(file_paths)
    
    print(f"Comparing {num_files} CSV files...")
    
    for i in range(num_files):
        for j in range(i + 1, num_files):
            file1, file2 = file_paths[i], file_paths[j]
            df1, df2 = all_dfs[file1], all_dfs[file2]
            if are_dataframes_identical(df1, df2):
                print(f"Identical files found: {file1} and {file2}")
            else:
                print(f"Files are different: {file1} and {file2}")

# Example usage
file_paths = [f'data/tableStandings/export ({num}).csv' for num in range(0, 15)]
check_csv_files_identical(file_paths)
