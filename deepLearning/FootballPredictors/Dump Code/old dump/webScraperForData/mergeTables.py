import pandas as pd

table1 = pd.read_csv('final.csv')
table2 = pd.read_csv('final_combined_data.csv')


table2 = table2.drop(columns=['Wk', 'Home', 'xG', 'xG.1', 'Away', 'Match Report', 'Result', 'Home_(\'Unnamed: 1_level_0\', \'#\')', 'Home_(\'Unnamed: 2_level_0\', \'Nation\')', 'Home_(\'Unnamed: 3_level_0\', \'Pos\')', 'Home_(\'Unnamed: 4_level_0\', \'Age\')', 'Away_(\'Unnamed: 1_level_0\', \'#\')', 'Away_(\'Unnamed: 2_level_0\', \'Nation\')', 'Away_(\'Unnamed: 3_level_0\', \'Pos\')', 'Away_(\'Unnamed: 4_level_0\', \'Age\')', 'Home_(\'Unnamed: 0_level_0\', \'Player\')', 'Away_(\'Unnamed: 0_level_0\', \'Player\')'])

combined_table = pd.concat([table1, table2], axis=1)

combined_table = combined_table.dropna(axis=0)

combined_table.to_csv('final_merged.csv')