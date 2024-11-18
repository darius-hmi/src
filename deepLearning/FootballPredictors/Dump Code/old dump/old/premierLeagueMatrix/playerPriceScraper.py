import requests
import pandas as pd
from bs4 import BeautifulSoup

# URL of the page to scrape
url = "https://fbref.com/en/comps/9/wages/Premier-League-Wages"

# Send a GET request to the URL
response = requests.get(url)

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# Find the table
table = soup.find('table', {'id': 'player_wages'})

# Read the table into a DataFrame
df = pd.read_html(str(table))[0]

# Save the DataFrame to a CSV file
df.to_csv('premier_league_wages.csv', index=False)

print("Data has been saved to premier_league_wages.csv")
