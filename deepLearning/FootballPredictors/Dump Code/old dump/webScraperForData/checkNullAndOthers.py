import pandas as pd


# df = pd.read_csv('cleanedDate.csv')

# null_columns = df.columns[df.isnull().any()]
# print(f"Columns with string null values are: {null_columns.tolist()}")

# string_columns = [col for col in df.columns if df[col].dtype == 'object']
# print(f"Columns with string values are : {string_columns}")

# # df_cleaned = df.dropna(axis=1)

# # df_cleaned.to_csv('cleanedDate.csv')



# Function to check for columns with identical names
def check_identical_names(df):
    col_names = df.columns.tolist()
    duplicates = [col for col in col_names if col_names.count(col) > 1]
    
    if duplicates:
        print("Columns with identical names:", duplicates)
    else:
        print("No identical column names found.")

# Function to check for columns with identical values throughout

def check_identical_columns(df):
    identical_columns = []
    # Compare each column with every other column
    for i, col1 in enumerate(df.columns):
        for j, col2 in enumerate(df.columns):
            if i < j:  # Avoid redundant comparisons and self-comparison
                if df[col1].equals(df[col2]):
                    identical_columns.append((col1, col2))

    if identical_columns:
        print("Columns with identical values across all rows:", identical_columns)
    else:
        print("No columns with identical values found.")



df = pd.read_csv('final.csv')

# Check for identical column names
check_identical_names(df)

# Check for columns with identical values throughout
check_identical_columns(df)
