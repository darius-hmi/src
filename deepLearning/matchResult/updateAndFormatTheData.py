import pandas as pd
import requests
from io import StringIO
import os


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


# URL of the webpage with tables
url1 = "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"
# Fetch the webpage
response = requests.get(url1)
# Wrap the HTML content in a StringIO object
html_content = StringIO(response.text)
# Read all the tables on the webpage
tables = pd.read_html(html_content)

if len(tables) == 1:
    table1 = tables[0]
    table1 = table1.drop(columns=['Day', 'Date', 'Time', 'Attendance', 'Venue', 'Referee', 'Match Report', 'Notes'])
    table1 = table1.dropna()
    table1 = table1.dropna(axis=1)


    if 'Score' in table1.columns:
        table1['Result'] = table1['Score'].apply(determine_result)
    
        # Reorder columns to place 'Result' next to 'Score'
        score_index = table1.columns.get_loc('Score')
        table1.insert(score_index + 1, 'Result', table1.pop('Result'))
        table1 = table1.drop(columns=['Score'])
        print('Score Table is saved.')

else:
    print('There are more than one table on the Scores Page.')





url2 = "https://fbref.com/en/comps/9/Premier-League-Stats"

response = requests.get(url2)
# Wrap the HTML content in a StringIO object
html_content = StringIO(response.text)
# Read all the tables on the webpage
tables = pd.read_html(html_content)

indices_to_keep = {0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22}

filtered_tables = [table for i, table in enumerate(tables) if i in indices_to_keep]

for i, table in enumerate(filtered_tables):
    if i != 0:
        table.columns = table.columns.droplevel(0)

filtered_tables[0] = filtered_tables[0].drop(columns=['Last 5', 'Attendance', 'Top Team Scorer', 'Goalkeeper', 'Notes'])
table_1_Columns = filtered_tables[1].columns
new_table_1_Columns = ['Rk', 'Squad'] + [f'H{col}' for col in table_1_Columns[2:14]] + [f'A{col}' for col in table_1_Columns[14:]]
filtered_tables[1].columns = new_table_1_Columns
if filtered_tables[2].columns[22] == 'Gls':
    filtered_tables[2].drop(filtered_tables[2].columns[22], axis=1, inplace=True)



for idx, table in enumerate(filtered_tables):
    # Drop multi-level headers if present
    # if isinstance(table.columns, pd.MultiIndex):
    #     table.columns = table.columns.droplevel(0)

    if idx != 0 and 'MP' in table.columns:
        table = table.drop(columns=['MP'])

    # Assign descriptive prefixes to the columns to avoid confusion
    home_stats = table.add_prefix(f'Home_')
    away_stats = table.add_prefix(f'Away_')

    # Merge home stats with fixtures
    table1 = table1.merge(home_stats, how='left', left_on='Home', right_on=f'Home_Squad', suffixes=('', f'_{idx}'))
    
    # Merge away stats with fixtures
    table1 = table1.merge(away_stats, how='left', left_on='Away', right_on=f'Away_Squad', suffixes=('', f'_{idx}'))
    
    # Drop unnecessary columns (like 'Home_Squad' and 'Away_Squad') after merging
    table1 = table1.drop(columns=[f'Home_Squad', f'Away_Squad'])

table1['Season'] = '2024/2025'


features = ['Home', 'Result', 'Away', 'Wk', 'xG', 'Home_HW', 'Home_HPts/MP', 
            'Home_HPts', 'Home_HGD', 'Home_HL', 'Home_Gls', 'Home_+/-90', 'xG.1',
            'Away_APts/MP', 'Away_AGD', 'Away_AL', 'Away_APts', 'Away_GA90', 'Away_AW', 
            'Away_L', 'Away_Pts', 'Away_PPM', 'Away_Pts/MP', 'Season']

table1 = table1[features]

# Save the filtered DataFrame to a new CSV file
table1.to_csv('filtered_updated.csv', index=False)
