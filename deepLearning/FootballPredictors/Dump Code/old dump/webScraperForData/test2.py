import pandas as pd
import requests
from bs4 import BeautifulSoup
import random
import time

base_url="https://fbref.com"


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

def extract_match_report_urls(soup, base_url):
    match_report_urls = []
    for row in soup.find_all('tr'):
        match_report_cell = row.find('td', text='Match Report')
        if match_report_cell:
            a_tag = match_report_cell.find('a')
            # Ensure we have a valid anchor tag and a URL
            if a_tag and 'href' in a_tag.attrs:
                # Prepend the base_url to the href
                match_report_urls.append(base_url + a_tag['href'])
            else:
                match_report_urls.append(None)
    return match_report_urls

# Fetch additional stats from match report URLs
def fetch_additional_stats(match_report_url):
    try:
        response = requests.get(match_report_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        tables = pd.read_html(response.text)

        if len(tables) >= 10:  # Ensure we have at least 11 tables
            last_row_table3 = tables[3].iloc[-1] if not tables[3].empty else None
            last_row_table10 = tables[10].iloc[-1] if not tables[10].empty else None
            
            if last_row_table3 is not None and last_row_table10 is not None:
                combined_stats = {**{f'Home_{col}': value for col, value in last_row_table3.items()},
                                  **{f'Away_{col}': value for col, value in last_row_table10.items()}}
                return combined_stats
    except Exception as e:
        print(f"Error fetching additional stats from '{match_report_url}': {e}")
        return None

# Fetch and process data including fixtures, stats, and additional match report stats
def fetch_and_process_data(season):
    # Construct the URL based on the season
    season_formatted = season.replace('/', '-')

    if season == '2024/2025':
        url1 = "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"
    else:
        season_formatted = season.replace('/', '-')
        url1 = f"https://fbref.com/en/comps/9/{season_formatted}/schedule/{season_formatted}-Premier-League-Scores-and-Fixtures"
    
    response = requests.get(url1)
    soup = BeautifulSoup(response.text, 'html.parser')
    tables = pd.read_html(response.text)

    if len(tables) >= 1:
        table1 = tables[0]
        table1 = table1.drop(columns=['Day', 'Date', 'Time', 'Attendance', 'Venue', 'Referee', 'Notes'], errors='ignore')
        table1 = table1.dropna().reset_index(drop=True)

        if 'Score' in table1.columns:
            table1['Result'] = table1['Score'].apply(determine_result)
            table1 = table1.drop(columns=['Score'])

        # Extract the Match Report URLs
        match_report_urls = extract_match_report_urls(soup, base_url)

        # Fetch additional stats and add them to the respective rows in table1
        for index, url in enumerate(match_report_urls):
            if url:
                time.sleep(random.uniform(6, 10))  # Avoid blocking
                combined_stats = fetch_additional_stats(url)
                if combined_stats:
                    # Add the combined stats to the current row in table1
                    for col, value in combined_stats.items():
                        table1.at[index, col] = value
            else:
                print(f'No valid URL for match at index {index}')

        return table1
    else:
        print(f"No valid tables found for season {season}.")
        return None

# List of seasons to process
seasons = ['2019/2020', '2020/2021', '2021/2022', '2022/2023', '2023/2024', '2024/2025']
all_tables = []

for season in seasons:
    time.sleep(random.uniform(6, 10))
    table = fetch_and_process_data(season)
    if table is not None:
        all_tables.append(table)

# Concatenate all season data into one table
final_table = pd.concat(all_tables, ignore_index=True)

# Rename any duplicate columns
final_table = rename_duplicates(final_table)

# Save to CSV
final_table.to_csv('final_combined_data.csv', index=False)
print('done.')
