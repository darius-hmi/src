import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../Data/final.csv')




#What are the chances that a team wins if they were leading at half time?
homeTeamsLeadingDataHT = data[data['home_team_goal_count_half_time'] > data['away_team_goal_count_half_time']]
homeTeamWinatFulltime = homeTeamsLeadingDataHT[homeTeamsLeadingDataHT['Result'] == 'H']
percentage_home_wins = (len(homeTeamWinatFulltime) / len(homeTeamsLeadingDataHT)) * 100 if len(homeTeamsLeadingDataHT) > 0 else 0

awayTeamsLeadingDataHT = data[data['home_team_goal_count_half_time'] < data['away_team_goal_count_half_time']]
awayTeamWinatFulltime = awayTeamsLeadingDataHT[awayTeamsLeadingDataHT['Result'] == 'A']
percentage_away_wins = (len(awayTeamWinatFulltime) / len(awayTeamsLeadingDataHT)) * 100 if len(awayTeamsLeadingDataHT) > 0 else 0

teamsLeadingDataHT = data[(data['home_team_goal_count_half_time'] > data['away_team_goal_count_half_time'])|
                          (data['home_team_goal_count_half_time'] < data['away_team_goal_count_half_time'])
                          ]
teamWinatFulltime = teamsLeadingDataHT[((teamsLeadingDataHT['home_team_goal_count_half_time'] > teamsLeadingDataHT['away_team_goal_count_half_time']) & (teamsLeadingDataHT['Result'] == 'H'))|
                                       ((teamsLeadingDataHT['home_team_goal_count_half_time'] < teamsLeadingDataHT['away_team_goal_count_half_time']) & (teamsLeadingDataHT['Result'] == 'A'))
                                       ]
percentage_all_wins = (len(teamWinatFulltime) / len(teamsLeadingDataHT)) * 100 if len(teamsLeadingDataHT) > 0 else 0




#what are the chanecs that the team not loses if they were leading at half time
teamsLeadingDataHTv2 = data[(data['home_team_goal_count_half_time'] > data['away_team_goal_count_half_time'])|
                          (data['home_team_goal_count_half_time'] < data['away_team_goal_count_half_time'])
                          ]
teamWinatFulltimev2 = teamsLeadingDataHTv2[((teamsLeadingDataHTv2['home_team_goal_count_half_time'] > teamsLeadingDataHTv2['away_team_goal_count_half_time']) & (teamsLeadingDataHTv2['Result'].isin(['H', 'D'])))|
                                       ((teamsLeadingDataHTv2['home_team_goal_count_half_time'] < teamsLeadingDataHTv2['away_team_goal_count_half_time']) & (teamsLeadingDataHTv2['Result'].isin(['A', 'D'])))
                                       ]
percentage_all_not_lose = (len(teamWinatFulltimev2) / len(teamsLeadingDataHTv2)) * 100 if len(teamsLeadingDataHTv2) > 0 else 0


labels = ['Home Team Leading', 'Away Team Leading', 'Any Team Leading']
percentages = [percentage_home_wins, percentage_away_wins, percentage_all_wins]


plt.bar(labels, percentages, color=['blue', 'orange', 'green'])
plt.title('Percentage of Halftime Leads Leading to Wins')
plt.ylabel('Percentage (%)')
plt.ylim(0, 100)

for i, v in enumerate(percentages):
    plt.text(i, v + 1, f"{v:.2f}%", ha='center', fontsize=10)

plt.show()






print(percentage_all_not_lose)
