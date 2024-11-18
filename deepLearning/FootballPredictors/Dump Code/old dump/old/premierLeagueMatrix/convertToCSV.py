import pandas as pd

# Load the XLSX file
df = pd.read_excel('data/playerWages/playerWages.xlsx', sheet_name='Sheet1')  # Adjust sheet_name if needed

# Save as CSV
df.to_csv('playerWages.csv', index=False)
