import pandas as pd
import requests
from io import StringIO
import os

# Rename any duplicated column names
def rename_duplicates(df):
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique(): 
        cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    return df

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
        print(f"Error processing score '{score}': {e}")
        return 'Invalid'

def fetch_and_process_data(season):
    # Determine the URLs based on the season
    if season == '2024/2025':
        url1 = "https://fbref.com/en/comps/16/schedule/League-Two-Scores-and-Fixtures"
        url2 = "https://fbref.com/en/comps/16/League-Two-Stats"
    else:
        # Modify URL for past seasons
        season_formatted = season.replace('/', '-')
        url1 = f"https://fbref.com/en/comps/16/{season_formatted}/schedule/{season_formatted}-League-Two-Scores-and-Fixtures"
        url2 = f"https://fbref.com/en/comps/16/{season_formatted}/{season_formatted}-League-Two-Stats"

    # Fetch and process the scores and fixtures
    response = requests.get(url1)
    html_content = StringIO(response.text)
    tables = pd.read_html(html_content)

    if len(tables) >= 1:
        if season == '2024/2025':
            table1 = tables[0]
        else:
            table1 = tables[1]
        table1 = table1.drop(columns=['Day', 'Date', 'Time', 'Attendance', 'Venue', 'Referee', 'Match Report', 'Notes'])
        table1 = table1.dropna()
        table1 = table1.dropna(axis=1)

        if 'Score' in table1.columns:
            table1['Result'] = table1['Score'].apply(determine_result)
            score_index = table1.columns.get_loc('Score')
            table1.insert(score_index + 1, 'Result', table1.pop('Result'))
            table1 = table1.drop(columns=['Score'])
        table1['Season'] = season
    else:
        print(f"There is not any table on the Scores Page for season {season}.")
        return None, None

    # Fetch and process the stats
    response = requests.get(url2)
    html_content = StringIO(response.text)
    tables = pd.read_html(html_content)

    indices_to_keep = {0, 1, 2, 4, 6, 8, 10}
    filtered_tables = [table for i, table in enumerate(tables) if i in indices_to_keep]

    for i, table in enumerate(filtered_tables):
        if i != 0:
            table.columns = table.columns.droplevel(0)

    columns_to_delete_table_0 = ['Last 5', 'Attendance', 'Top Team Scorer', 'Goalkeeper', 'Notes']
    filtered_tables[0] = filtered_tables[0].drop(columns=[col for col in columns_to_delete_table_0 if col in filtered_tables[0].columns], errors='ignore')
    table_1_Columns = filtered_tables[1].columns
    new_table_1_Columns = ['Rk', 'Squad'] + [f'H{col}' for col in table_1_Columns[2:10]] + [f'A{col}' for col in table_1_Columns[10:]]
    filtered_tables[1].columns = new_table_1_Columns

    combined_table = pd.DataFrame()

    for table in filtered_tables:
        # columns_to_delete = ['90s', 'Starts', '# Pl', 'Min']
        # table = table.drop(columns=[col for col in columns_to_delete if col in table.columns], errors='ignore')
        
        if 'Squad' not in table.columns:
            raise ValueError("'Squad' column is missing from one of the tables.")
        
        if combined_table.empty:
            combined_table = table
        else:
            columns_to_add = [col for col in table.columns if col not in combined_table.columns]
            if columns_to_add:
                combined_table = combined_table.merge(table[['Squad'] + columns_to_add], on='Squad', how='left')

    home_stats = combined_table.add_prefix(f'Home_')
    away_stats = combined_table.add_prefix(f'Away_')

    # Merge home and away stats with fixtures
    table1 = table1.merge(home_stats, how='left', left_on='Home', right_on='Home_Squad')
    table1 = table1.merge(away_stats, how='left', left_on='Away', right_on='Away_Squad')

    # Drop unnecessary columns (like 'Home_Squad' and 'Away_Squad') after merging
    table1 = table1.drop(columns=[f'Home_Squad', f'Away_Squad'])

    return table1

# List of seasons to process
seasons = ['2019/2020', '2020/2021', '2021/2022', '2022/2023', '2023/2024', '2024/2025']

all_tables = []

for season in seasons:
    table = fetch_and_process_data(season)
    if table is not None:
        all_tables.append(table)

# Concatenate all season data into one table
final_table = pd.concat(all_tables, ignore_index=True)
final_table = final_table.dropna(axis=1)
final_table = rename_duplicates(final_table)
final_table.to_csv('final.csv', index=False)
print('done.')
