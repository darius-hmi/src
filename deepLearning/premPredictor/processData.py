import pandas as pd
import os

# Directory containing your additional CSV files
data_dir = 'data/2022-2023/'

# Load the base CSV file
base_file = 'data/2022-2023/table_1.csv'
base_df = pd.read_csv(base_file)

# List of additional CSV files to check
additional_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and f != 'table_1.csv']

# Process each additional CSV file
for file_name in additional_files:
    file_path = os.path.join(data_dir, file_name)
    df = pd.read_csv(file_path)
    
    # Iterate over each squad in the base DataFrame
    for squad in base_df['Squad']:
        if squad in df['Squad'].values:
            squad_data = df[df['Squad'] == squad]
            
            # Drop the 'Squad' column if it's present in the additional file
            squad_data = squad_data.drop(columns='Squad', errors='ignore')
            
            # Iterate over columns in the additional DataFrame
            for col in squad_data.columns:
                if col not in base_df.columns:
                    # Add new columns to base_df if they do not exist
                    base_df[col] = pd.NA
                
                # Update base_df with data from the additional file
                base_df.loc[base_df['Squad'] == squad, col] = squad_data[col].values

# Save the updated base DataFrame to a new CSV file
base_df.to_csv('data/2022-2023/updated_base_data.csv', index=False)

print("Base data updated and saved to 'data/updated_base_data.csv'.")
