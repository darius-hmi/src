import pandas as pd


match_results = pd.read_csv('data/combined_fixtures_with_results.csv')

match_results = match_results.drop(columns=['Day','Date','Time','Score','Referee', 'Home_% Estimated', 'Away_% Estimated'])


# Define a function to extract the value
def extract_currency_value(text, currency='£'):
    # Split the text to isolate the currency values
    parts = text.split(' ')
    
    # Find the index of the currency you want (in this case £)
    for i, part in enumerate(parts):
        if part.startswith(currency):
            # Remove the currency symbol and commas, then convert to integer
            value_str = parts[i + 1].replace(',', '')
            return int(value_str)

def percentage_to_integer(percentage_str):
    # Remove the '%' sign and convert to float
    numeric_value = float(percentage_str.replace('%', '').strip())
    # Convert to integer
    return int(numeric_value)


# Apply the function to the 'Home_Weekly Wages' column
match_results['Home_Weekly_Wages_in_GBP'] = match_results['Home_Weekly Wages'].apply(lambda x: extract_currency_value(x, currency='£'))
match_results['Home_Annual_Wages_in_GBP'] = match_results['Home_Annual Wages'].apply(lambda x: extract_currency_value(x, currency='£'))
match_results['Away_Weekly_Wages_in_GBP'] = match_results['Away_Weekly Wages'].apply(lambda x: extract_currency_value(x, currency='£'))
match_results['Away_Annual_Wages_in_GBP'] = match_results['Away_Annual Wages'].apply(lambda x: extract_currency_value(x, currency='£'))


match_results = match_results.drop(columns=['Home_Weekly Wages', 'Home_Annual Wages', 'Away_Weekly Wages', 'Away_Annual Wages'])


columns_with_null = match_results.columns[match_results.isnull().any()]

print("Old DataFrame shape:", match_results.shape)

df_cleaned = match_results.dropna(axis=1)

print("New DataFrame shape:", df_cleaned.shape)

# Now you have a new column with just the integer values
print("Columns with null values:")
print(columns_with_null)


df_cleaned.to_csv('cleaned_file_noNull_no_noNonInteger.csv', index=False)