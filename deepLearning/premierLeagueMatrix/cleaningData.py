import pandas as pd

# Load the datasets
match_results = pd.read_csv('data/matchResults/combinedMatchResults.csv')
standings = pd.read_csv('data/tableStandings/combinedTables.csv')
player_stats = pd.read_csv('data/playerStats/playerStats.csv')
player_wages = pd.read_csv('data/playerWages/playerWages.csv')


match_results = match_results.dropna(how='all')
match_results = match_results.drop(columns=['BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA']) 

match_results['Date'] = pd.to_datetime(match_results['Date'], format='%d/%m/%Y')

def date_to_season(date):
    # Get the year from the date
    year = date.year
    
    # Determine if the date falls between July 1st of the current year and June 30th of the next year
    if date >= pd.Timestamp(year=year, month=7, day=1) and date <= pd.Timestamp(year=year+1, month=6, day=30):
        return f"{year}/{year + 1}"
    else:
        return f"{year - 1}/{year}"

# Apply the function to the date column
match_results['season'] = match_results['Date'].apply(date_to_season)


standings = standings.dropna(how='any')


player_stats = player_stats.dropna(subset=['Name'])
player_stats_cleaned = player_stats.dropna(subset=['Weight(kg)', 'Height(cm)', 'Age'])
average_age = player_stats_cleaned['Age'].mean()
average_wight = player_stats_cleaned['Weight(kg)'].mean()
average_height = player_stats_cleaned['Height(cm)'].mean()
player_stats = player_stats.fillna({'Age': average_age, 'Weight(kg)':average_wight, 'Height(cm)':average_height, 'Jersey':'5', 'Citizenship':'England'})
player_stats = player_stats.drop(columns=['Id', 'Update Time']) 


player_wages = player_wages.drop(columns=['Nation', 'Pos', 'Age', 'Annual Wages', 'Notes', 'Rk']) 



print("Missing values in Match Results:")
print(match_results.isnull().sum())

print("\nMissing values in Table Standings:")
print(standings.isnull().sum())

print("\nMissing values in Player Stats:")
print(player_stats.isnull().sum())

print("\nMissing values in Player Wages:")
print(player_wages.isnull().sum())


match_results.to_csv('cleaned_match_results.csv', index=False)
standings.to_csv('cleaned_table_standings.csv', index=False)
player_stats.to_csv('cleaned_player_stats.csv', index=False)
player_wages.to_csv('cleaned_player_wages.csv', index=False)