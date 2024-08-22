import pandas as pd
import os

# Directory where your CSV files are located
directory = 'data/2023-2024'

# Loop through table_5.csv to table_24.csv
for i in range(3, 25):
    filename = f'table_{i}.csv'
    file_path = os.path.join(directory, filename)
    
    # Check if the file exists
    if os.path.exists(file_path):
        # Read the CSV file, skipping the first row
        df = pd.read_csv(file_path, skiprows=1)
        
        # Save the dataframe back to the CSV file
        df.to_csv(file_path, index=False)
        print(f'Processed {filename}')
    else:
        print(f'{filename} does not exist')

print('All files processed.')
