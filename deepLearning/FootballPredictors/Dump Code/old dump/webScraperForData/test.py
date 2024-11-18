import pandas as pd
import requests
from bs4 import BeautifulSoup
import random
import time

# Rename any duplicated column names
def rename_duplicates(df):
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique(): 
        cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]  
    df.columns = cols
    return df

def determine_result(score):
    try:
        score = score.replace('–', '-').replace('—', '-').replace('−', '-')
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

# Extract the first Match Report URL from the HTML soup
def extract_first_match_report_url(soup, base_url):
    for row in soup.find_all('tr'):
        match_report_cell = row.find('td', text='Match Report')
        if match_report_cell:
            a_tag = match_report_cell.find('a')
            if a_tag and 'href' in a_tag.attrs:
                return base_url + a_tag['href']
    return None

# Fetch additional stats (table3 and table10) from the first match report URL
def fetch_additional_stats(match_report_url, headers):
    try:
        response = requests.get(match_report_url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        tables = pd.read_html(response.text)
        
        if len(tables) >= 10:
            table3 = tables[3]
            table3.columns = table3.columns.droplevel(0)
            table10 = tables[10]
            table10.columns = table10.columns.droplevel(0)

            last_row_table3 = table3.iloc[-1] if not table3.empty else None
            last_row_table10 = table10.iloc[-1] if not table10.empty else None

            return last_row_table3, last_row_table10
    except Exception as e:
        print(f"Error fetching additional stats from '{match_report_url}': {e}")
        return None, None

# Fetch and process data for each season
def fetch_and_process_data(season, headers, base_url="https://fbref.com"):
    # Determine URLs based on the season
    if season == '2024/2025':
        url1 = "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"
        url2 = "https://fbref.com/en/comps/9/Premier-League-Stats"
    else:
        season_formatted = season.replace('/', '-')
        url1 = f"https://fbref.com/en/comps/9/{season_formatted}/schedule/{season_formatted}-Premier-League-Scores-and-Fixtures"
        url2 = f"https://fbref.com/en/comps/9/{season_formatted}/{season_formatted}-Premier-League-Stats"

    # Fetch fixtures table (table1)
    response = requests.get(url1, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    tables = pd.read_html(response.text)

    if len(tables) >= 1:
        table1 = tables[0]
        table1 = table1.drop(columns=['Day', 'Date', 'Time', 'Attendance', 'Venue', 'Referee', 'Notes'])
        table1 = table1.dropna()
        table1 = table1.dropna(axis=1)

        if 'Score' in table1.columns:
            table1['Result'] = table1['Score'].apply(determine_result)
            score_index = table1.columns.get_loc('Score')
            table1.insert(score_index + 1, 'Result', table1.pop('Result'))
            table1 = table1.drop(columns=['Score'])
        table1['Season'] = season
    else:
        print(f"No valid tables found for season {season}.")
        return None

    # Fetch the first Match Report URL
    first_match_report_url = extract_first_match_report_url(soup, base_url)
    additional_stats = []



    if first_match_report_url:
        time.sleep(random.uniform(6, 10))  # Add delay to avoid blocking
        last_row_table3, last_row_table10 = fetch_additional_stats(first_match_report_url, headers)
        if last_row_table3 is not None and last_row_table10 is not None:
            home_stats = {f'Home_thisGame_{col}': value for col, value in last_row_table3.items()}
            away_stats = {f'Away_thisGame_{col}': value for col, value in last_row_table10.items()}
            combined_stats = {**home_stats, **away_stats}
            additional_stats.append(combined_stats)
    else:
        print(f"No match report URL found for season {season}")

    # Add additional stats to the first row of table1
    if additional_stats:
        additional_stats_df = pd.DataFrame(additional_stats)
        # Ensure there is only one row in additional_stats_df
        if additional_stats_df.shape[0] == 1:
            # Convert additional_stats_df to a Series
            additional_stats_row = additional_stats_df.iloc[0]
            
            # Add the additional stats to the first row of table1
            for col in additional_stats_row.index:
                if col not in table1.columns:
                    # If the column doesn't exist, add it
                    table1[col] = None  # Initialize with None
                table1.at[0, col] = additional_stats_row[col]  # Set the value in the first row
        else:
            print("Expected one row in additional_stats_df, found:", additional_stats_df.shape[0])


    # Fetch and process stats page (table2)
    response = requests.get(url2, headers=headers)
    tables = pd.read_html(response.text)

    indices_to_keep = {0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22}
    filtered_tables = [table for i, table in enumerate(tables) if i in indices_to_keep]

    for i, table in enumerate(filtered_tables):
        if i != 0:
            table.columns = table.columns.droplevel(0)

    combined_table = pd.DataFrame()
    for table in filtered_tables:
        columns_to_delete = ['90s', 'Starts', '# Pl', 'Min']
        table = table.drop(columns=[col for col in columns_to_delete if col in table.columns], errors='ignore')
        
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

    # Drop unnecessary columns
    table1 = table1.drop(columns=[f'Home_Squad', f'Away_Squad'], errors='ignore')

    # Ensure the index is unique
    table1.reset_index(drop=True, inplace=True)

    return table1

# List of seasons to process
seasons = ['2019/2020', '2020/2021', '2021/2022', '2022/2023', '2023/2024', '2024/2025']
all_tables = []

# Headers to avoid being blocked as a bot
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

# Fetch and process data for all seasons
for season in seasons:
    time.sleep(random.uniform(6, 10))
    table = fetch_and_process_data(season, headers=headers)
    if table is not None:
        all_tables.append(table.reset_index(drop=True))  # Reset index here

# Check if there are tables to concatenate
if all_tables:
    # Concatenate all season data into one table
    final_table = pd.concat(all_tables, ignore_index=True)
    # Save to CSV
    final_table.to_csv('final_combined_data.csv', index=False)
    print('done.')
else:
    print('No data to concatenate.')


for idx, df in enumerate(all_tables):
    if not df.index.is_unique:
        print(f"DataFrame at index {idx} has duplicate indices.")

# Proceed with concatenation after fixing any issues