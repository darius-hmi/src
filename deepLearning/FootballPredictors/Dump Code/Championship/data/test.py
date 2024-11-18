import pandas as pd
import requests
from io import StringIO
import os, time, random


url1 = "https://fbref.com/en/comps/10/2023-2024/schedule/2023-2024-Championship-Scores-and-Fixtures"


response = requests.get(url1)
html_content = StringIO(response.text)
tables = pd.read_html(html_content)


table1 = tables[0]

table1.to_csv('test.csv', index=False)