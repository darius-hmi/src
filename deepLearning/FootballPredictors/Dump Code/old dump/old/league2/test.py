import pandas as pd
import requests

url1 = "https://fbref.com/en/comps/16/2023-2024/2023-2024-League-Two-Stats"

# Fetch and process the scores and fixtures
response = requests.get(url1)
html_content = response.text

# Parse the HTML content into tables
tables = pd.read_html(html_content)

# Loop through each table and print the head
for i, table in enumerate(tables):
    print(f"Table {i}:\n")
    print(table.head(), "\n")
