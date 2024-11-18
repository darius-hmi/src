import pandas as pd


match_results = pd.read_csv('temp.csv')

null_counts = match_results.isnull().sum()
print("Null values count in each column:")
print(null_counts)

# Display rows with any null values
rows_with_nulls = match_results[match_results.isnull().any(axis=1)]
print("\nRows with null values:")
print(rows_with_nulls)






# match_results = pd.read_csv('cleaned_match_results.csv')

# # Example mapping dictionary (customize this based on your needs)
# team_name_mapping = {
#     'QPR': 'Queens Park Rangers',
#     'Wigan': 'Wigan Athletic',
#     'Nott\'m Forest': 'Nottingham Forest',
#     'Birmingham': 'Birmingham City',
#     'Hull': 'Hull City',
#     'Man City': 'Manchester City',
#     'Norwich': 'Norwich City',
#     'Newcastle': 'Newcastle United',
#     'Cardiff': 'Cardiff City',
#     'Blackburn': 'Blackburn Rovers',
#     'Stoke': 'Stoke City',
#     'Leicester': 'Leicester City',
#     'Man United': 'Manchester United',
#     'Brighton': 'Brighton & Hove Albion',
#     'Tottenham': 'Tottenham Hotspur',
#     'Wolves': 'Wolverhampton Wanderers',
#     'Bournemouth': 'AFC Bournemouth',
#     'Swansea': 'Swansea City',
#     'West Brom': 'West Bromwich Albion',
#     'Luton': 'Luton Town',
#     'Leeds': 'Leeds United',
#     'Huddersfield': 'Huddersfield Town',
#     'Bolton': 'Bolton Wanderers',
#     'West Ham': 'West Ham United',
#     # Add more mappings as necessary
# }

# # Replace team names in the DataFrame
# match_results['AwayTeam'] = match_results['AwayTeam'].replace(team_name_mapping)
# match_results['HomeTeam'] = match_results['HomeTeam'].replace(team_name_mapping)

# # Optionally, save the updated DataFrame back to a CSV file
# match_results.to_csv('updated_match_results.csv', index=False)

# # Display the updated DataFrame to verify changes
# print(match_results.head())











# match_results = pd.read_csv('cleaned_match_results.csv')
# standings = pd.read_csv('cleaned_table_standings.csv')

# # Extract unique team names
# match_teams = set(match_results['AwayTeam'].unique()) | set(match_results['HomeTeam'].unique())
# standings_teams = set(standings['Team'].unique())

# # Find teams in match_results that are not in standings
# missing_teams = match_teams - standings_teams
# print("Teams in match_results but not in standings:", missing_teams)





# standings = pd.read_csv('data/cleaned_table_standings.csv')

# def year_to_season(year):
#     next_year = year + 1
#     return f"{year}/{next_year}"  # Format as '2010/2011'

# standings['season'] = standings['Year'].apply(year_to_season)

# standings.drop(columns=['Year'], inplace=True)

# standings.to_csv('cleaned_table_standings.csv', index=False)

# print(standings)
