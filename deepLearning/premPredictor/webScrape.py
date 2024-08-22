import pandas as pd
import requests
from io import StringIO
import os

# URL of the webpage with tables
url = "https://fbref.com/en/comps/9/2019-2020/wages/2019-2020-Premier-League-Wages"

# Fetch the webpage
response = requests.get(url)

# Wrap the HTML content in a StringIO object
html_content = StringIO(response.text)

# Read all the tables on the webpage
tables = pd.read_html(html_content)

# Specify the directory where you want to save the CSV files
save_directory = 'data/2019-2020'

# Create the directory if it doesn't exist
os.makedirs(save_directory, exist_ok=True)

# Save each table to a CSV file in the specified directory
for i, table in enumerate(tables):
    file_path = os.path.join(save_directory, f'table_{i+26}.csv')
    table.to_csv(file_path, index=False)

print(f"Tables saved to {save_directory}")
