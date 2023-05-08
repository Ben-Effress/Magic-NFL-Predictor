import requests
from bs4 import BeautifulSoup
import pandas as pd
import os.path
import math
import numpy as np
import csv

# Set the initial Elo rating for each team
initial_elo_rating = 1500

# Define a function to calculate the expected win probability for a team
def expected_win_probability(team_elo_rating, opponent_elo_rating):
    return 1 / (1 + math.pow(10, (opponent_elo_rating - team_elo_rating) / 400))

filename = 'nfl_scores.csv'

if not os.path.exists(filename):
    # Create a list to hold the DataFrames for each week
    dataframes = []

    # Create a dictionary to hold the current Elo ratings for each team
    elo_ratings = {}

    for year in range(2020, 2023):
        # Loop through each week of the season
        for week in range(1, 18):
            print(week)
            url = "https://www.espn.com/nfl/schedule/_/week/" + str(week) + "/year/" + str(year) + "/seasontype/2"
            response = requests.get(url)

            soup = BeautifulSoup(response.content, 'html.parser')

            # Find all HTML elements with class "teams__col"
            team_elements = soup.findAll(class_='teams__col')

            # Create a list to hold the data for each game in the week
            week_data = []

            # Loop through each team element and extract the team names and scores
            for team_element in team_elements:
                # Find the child HTML element with tag "a"
                link_element = team_element.find('a')

                # Extract the text content of the child element
                text = link_element.text.strip()

                # Split the text content into a list of team names and scores
                parts = text.split(', ')

                # Extract the team names and scores from the list
                if text != "Canceled" and text != "Postponed":
                    team1, score1 = parts[0].split(' ')[0], int(parts[0].split(' ')[1])
                    team2, score2 = parts[1].split(' ')[0], int(parts[1].split(' ')[1])

                    # Calculate the expected win probability for each team
                    team1_elo_rating = elo_ratings.get(team1, initial_elo_rating)
                    team2_elo_rating = elo_ratings.get(team2, initial_elo_rating)
                    team1_expected_win_probability = expected_win_probability(team1_elo_rating, team2_elo_rating)
                    team2_expected_win_probability = 1 - team1_expected_win_probability

                    # Update the Elo ratings for each team based on the actual outcome of the game
                    if score1 > score2:
                        team1_actual_win_probability = 1
                        team2_actual_win_probability = 0
                    else:
                        team1_actual_win_probability = 0
                        team2_actual_win_probability = 1
                        
                    K = 50 - (0.6 * week) + (20 * (year - 2020))
                    print(K)
                    
                    team1_new_elo_rating = team1_elo_rating + K * (team1_actual_win_probability - team1_expected_win_probability)
                    team2_new_elo_rating = team2_elo_rating + K * (team2_actual_win_probability - team2_expected_win_probability)

                    elo_ratings[team1] = team1_new_elo_rating
                    elo_ratings[team2] = team2_new_elo_rating

                    # Append the data as a new row to the week data list
                    week_data.append([team1, score1, team2, score2, team1_elo_rating, team2_elo_rating, team1_new_elo_rating, team2_new_elo_rating, week, year])
                    print(team1, score1, team2, score2, team1_elo_rating, team2_elo_rating, team1_new_elo_rating, team2_new_elo_rating)

            # Convert the week data list to a DataFrame and append it to the dataframes list
            week_dataframe = pd.DataFrame(week_data, columns=['Team 1', 'Score 1', 'Team 2', 'Score 2', 'Elo Rating 1', 'Elo Rating 2', 'New Elo Rating 1', 'New Elo Rating 2', 'Week', 'Year'])
            dataframes.append(week_dataframe)

    # Concatenate all the DataFrames vertically into one DataFrame
    data = pd.concat(dataframes, ignore_index=True)

    # Save the data to a CSV file
    data.to_csv(filename, index=False)

    # Print a message to indicate that the file has been saved
    print('Data saved to', filename)
    
    # Sort the elo_ratings dictionary by Elo rating
    sorted_elo_ratings = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
    
    # Dictionary of team names and truncated Elo ratings
    team_elo_ratings = {}

    # Loop through the sorted elo_ratings list
    for team, elo_rating in sorted_elo_ratings:
        # Truncate the Elo rating to the first 4 digits
        truncated_elo_rating = str(elo_rating)[:4]
        
        # Give team elo ratings to dictionary
        team_elo_ratings[team] = truncated_elo_rating
        
        # Print the team name and truncated Elo rating
        print(team, truncated_elo_rating)
    
    # Save the team_elo_ratings dictionary to a CSV file
    ratings_filename = 'team_elo_ratings.csv'
    ratings_data = pd.DataFrame.from_dict(team_elo_ratings, orient='index', columns=['Elo Rating'])
    ratings_data.index.name = 'Team'
    ratings_data.to_csv(ratings_filename)

    
else:
    # Load the data from the CSV file
    data = pd.read_csv(filename)

    # Print the data
    print(data)
    
    
week_lower = 15
week_upper = 15.4
year_lower = 145
year_upper = 155
week_step = .005
year_step = .01
week_values = np.arange(week_lower, week_upper, week_step)
year_values = np.arange(year_lower, year_upper, year_step)
count = 0
total = (week_upper - week_lower) * (year_upper - year_lower) * (1/week_step) * (1/year_step)
print(total)
import time
start_time = time.time()

least_squares_dict = {}
least_squares_qb_dict = {}
ls_elo_dict = {}
ls_qb_elo_dict = {}

min_least_squared = [0,0,10000000000000000000000000]
min_least_squared_qb = [0,0,10000000000000000000000000]
min_ls_elo = [0,0,10000000000000000000000000]
min_ls_qb_elo = [0,0,10000000000000000000000000]

best_week_value = 15.3
best_year_value = 153.5

# best_week_value = 0
# best_year_value = 4.1

#QBRANK: 15.659999999999986 150.0 1318

for week_coeff in week_values:
    for year_coeff in year_values:
# for week_coeff in [best_week_value]:
#     for year_coeff in [best_year_value]:
        count += 1
        percent_done = count/total
        
        #estimate remaining time
        time_elapsed = time.time() - start_time
        time_per_iteration = time_elapsed/count
        remaining_iterations = total - count
        remaining_time = remaining_iterations * time_per_iteration
        #print remaining time
        # print("Time left: {:.2f} seconds".format(remaining_time))
        
        if week_coeff%.1 == 0:
            print("Percent done: {:.4%}".format(percent_done))
        # Create a dictionary to hold the current Elo ratings for each team
        elo_ratings = {}
        
        game_data = pd.read_csv('game_data.csv')
        
        with open('game_data.csv') as game_data:
            reader = csv.reader(game_data)
            for row in reader:
                team1, score1, team2, score2, week, year = row
                # print(team1, score1, team2, score2, week, year)
                
                # Calculate the expected win probability for each team
                team1_elo_rating = elo_ratings.get(team1, initial_elo_rating)
                team2_elo_rating = elo_ratings.get(team2, initial_elo_rating)
                team1_expected_win_probability = expected_win_probability(team1_elo_rating, team2_elo_rating)
                team2_expected_win_probability = 1 - team1_expected_win_probability
                
                # Update the Elo ratings for each team based on the actual outcome of the game
                if score1 > score2:
                    team1_actual_win_probability = 1
                    team2_actual_win_probability = 0
                else:
                    team1_actual_win_probability = 0
                    team2_actual_win_probability = 1
                        
                K = 32 - (week_coeff * int(week)) + (year_coeff * (int(year) - 2020))
                # print(K)
                
                team1_new_elo_rating = team1_elo_rating + K * (team1_actual_win_probability - team1_expected_win_probability)
                team2_new_elo_rating = team2_elo_rating + K * (team2_actual_win_probability - team2_expected_win_probability)

                elo_ratings[team1] = team1_new_elo_rating
                elo_ratings[team2] = team2_new_elo_rating
                # print(elo_ratings)
                
            # Sort the elo_ratings dictionary by Elo rating
            sorted_elo_ratings = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
                
            # print(sorted_elo_ratings)
                
            # # Dictionary of team names and truncated Elo ratings
            team_elo_ratings = {}

            # Loop through the sorted elo_ratings list
            for team, elo_rating in sorted_elo_ratings:
                # Truncate the Elo rating to the first 4 digits
                truncated_elo_rating = float(str(elo_rating)[:4])

                # Give team Elo ratings to dictionary
                team_elo_ratings[team] = truncated_elo_rating

                # Print the team name and truncated Elo rating
                # print(team, truncated_elo_rating)

            # Create a DataFrame from the team Elo ratings dictionary
            predicted_elo_ratings = pd.DataFrame.from_dict(team_elo_ratings, orient='index', columns=['Elo Rating'])
            
            #rename Elo Rating to Predicted Elo Rating
            predicted_elo_ratings.rename(columns={'Elo Rating': 'Predicted Elo Rating'}, inplace=True)
           
            # filename = 'elo_ratings.csv'
            # predicted_elo_ratings.index.name = 'Team'
            # predicted_elo_ratings.to_csv(filename)

            # Set the name of the index column
            predicted_elo_ratings.index.name = 'Team'
            # print(predicted_elo_ratings.info())
            

            # Add the "Predicted Rank" column to the DataFrame
            predicted_elo_ratings["Predicted Rank"] = range(1, 32 + 1)
            # print(predicted_elo_ratings)
                
            real_elo_ratings = pd.read_csv('ranked_elo_ratings.csv')
            real_elo_ratings["Real Rank"] = range(1, 32 + 1)
            
            real_qb_adjusted_elo_ratings = pd.read_csv('ranked_qb_adjusted_elo_ratings.csv')
            real_qb_adjusted_elo_ratings["Real QB Adjusted Rank"] = range(1, 32 + 1)
                
            merged_df = pd.merge(real_elo_ratings, real_qb_adjusted_elo_ratings, on='Team').merge(predicted_elo_ratings, on='Team')
            merged_df = merged_df[['Team', 'Predicted Rank', 'Real Rank', 'Real QB Adjusted Rank', 'Predicted Elo Rating', 'Real Elo Rating', 'QB Adjusted Elo Rating']]
            # print(merged_df.info())
            merged_df["Rank Difference"] = merged_df["Real Rank"] - merged_df["Predicted Rank"]
            merged_df["QB Rank Difference"] = merged_df["Real QB Adjusted Rank"] - merged_df["Predicted Rank"]
            merged_df["Elo Difference"] = merged_df["Real Elo Rating"] - merged_df["Predicted Elo Rating"]
            merged_df["QB Elo Difference"] = merged_df["QB Adjusted Elo Rating"] - merged_df["Predicted Elo Rating"]
            # print(merged_df)
            least_squares_real = merged_df[["Team", "Rank Difference"]]
            least_squares_real_qb = merged_df[["Team", "QB Rank Difference"]]
            ls_elo = merged_df[["Team", "Elo Difference"]]
            ls_qb_elo = merged_df[["Team", "QB Elo Difference"]]
            pd.options.mode.chained_assignment = None
            # print(least_squares)
            least_squares_real["Squared Difference"] = least_squares_real["Rank Difference"] ** 2
            least_squares_real_qb["Squared Difference"] = least_squares_real_qb["QB Rank Difference"] ** 2
            ls_elo["Squared Difference"] = ls_elo["Elo Difference"] ** 2
            ls_qb_elo["Squared Difference"] = ls_qb_elo["QB Elo Difference"] ** 2
            # print(ls_elo)

            # print(least_squares)
            least_squares_real_sum_rank = least_squares_real["Squared Difference"].sum()
            least_squares_real_qb_sum_rank = least_squares_real_qb["Squared Difference"].sum()
            ls_elo_sum = ls_elo["Squared Difference"].sum()
            ls_qb_elo_sum = ls_qb_elo["Squared Difference"].sum()
            
            # print(least_sqaures_sum)
            least_squares_dict[tuple([week_coeff, year_coeff])] = least_squares_real_sum_rank
            least_squares_qb_dict[tuple([week_coeff, year_coeff])] = least_squares_real_qb_sum_rank
            ls_elo_dict[tuple([week_coeff, year_coeff])] = ls_elo_sum
            ls_qb_elo_dict[tuple([week_coeff, year_coeff])] = ls_qb_elo_sum
            
            
            if least_squares_real_sum_rank < min_least_squared[2]:
                min_least_squared = [week_coeff, year_coeff, least_squares_real_sum_rank]
                print("REAL RANK: " + str(min_least_squared))
            if least_squares_real_qb_sum_rank < min_least_squared_qb[2]:
                min_least_squared_qb = [week_coeff, year_coeff, least_squares_real_qb_sum_rank]
                print("QB RANK: " + str(min_least_squared))
            if ls_elo_sum < min_ls_elo[2]:
                min_ls_elo = [week_coeff, year_coeff, ls_elo_sum]
                print("ELO: " + str(min_ls_elo))
            if ls_qb_elo_sum < min_ls_qb_elo[2]:
                min_ls_qb_elo = [week_coeff, year_coeff, ls_qb_elo_sum]
                print("QB ELO: " + str(min_ls_qb_elo))
            
            
# Find the tuple with the smallest least squared sum
min_tuple = min(least_squares_dict, key=least_squares_dict.get)
# for key, value in least_squares_dict.items():
#     print("LS RANK: " + f'Week Coefficient: {key[0]}, Year Coefficient: {key[1]}, Least Squared Sum: {value}')
    
min_tuple_qb = min(least_squares_qb_dict, key=least_squares_qb_dict.get)
# for key, value in least_squares_qb_dict.items():
#     print("LS QB RANK: " + f'Week Coefficient: {key[0]}, Year Coefficient: {key[1]}, Least Squared Sum: {value}')
    
min_tuple_elo = min(ls_elo_dict, key=ls_elo_dict.get)
# for key, value in ls_elo_dict.items():
#     print("LS ELO: " + f'Week Coefficient: {key[0]}, Year Coefficient: {key[1]}, Least Squared Sum: {value}')
    
min_tuple_qb_elo = min(ls_qb_elo_dict, key=ls_qb_elo_dict.get)
# for key, value in ls_qb_elo_dict.items():
#     print("LS QB RANK: " + f'Week Coefficient: {key[0]}, Year Coefficient: {key[1]}, Least Squared Sum: {value}')
    
# Get the week coefficient and year coefficient from the tuple
week_coeff, year_coeff = min_tuple
min_least_squared_value = least_squares_dict[min_tuple]
print(week_coeff, year_coeff, min_least_squared_value)

# Get the week coefficient and year coefficient from the tuple
week_coeff, year_coeff = min_tuple_qb
min_least_squared_value = least_squares_qb_dict[min_tuple]
print(week_coeff, year_coeff, min_least_squared_value)

# Get the week coefficient and year coefficient from the tuple
week_coeff, year_coeff = min_tuple_elo
min_least_squared_value = ls_elo_dict[min_tuple]
print(week_coeff, year_coeff, min_least_squared_value)

# Get the week coefficient and year coefficient from the tuple
week_coeff, year_coeff = min_tuple_qb_elo
min_least_squared_value = ls_qb_elo_dict[min_tuple]
print(week_coeff, year_coeff, min_least_squared_value)

end_time = time.time()

total_time = end_time - start_time

print("Total execution time:", total_time, "seconds")
print(total)
print(total_time/total)
print((5*60*60)/(total_time/total))
print(((5*60*60)/(total_time/total))**.5)
print((((5*60*60)/(total_time/total))**.5) *.05)

# 4.32 43.5 1410
# 15.38 147.02 1318
# 0.4599999999994573 7.5 138911.0
# 1.6399999999994321 14.5 175672.0

                
                
        
        
        
        
    
    
# real_power_rankings = pd.read_csv('power_rankings_end_of_2022.csv')


# guessed_team_elo_ratings = pd.read_csv('team_elo_ratings.csv')
# guessed_team_elo_ratings["Predicted Rank"] = range(1, 32 + 1)
# print(guessed_team_elo_ratings)

# merged_df = pd.merge(real_power_rankings, guessed_team_elo_ratings, on='Team')
# merged_df = merged_df[['Team', 'Real Rank', 'Predicted Rank']]
# print(merged_df)
# merged_df["Rank Difference"] = merged_df["Real Rank"] - merged_df["Predicted Rank"]

# least_squares = merged_df[["Team", "Rank Difference"]]
# print(least_squares)
# least_squares["Squared Difference"] = least_squares["Rank Difference"] ** 2

# print(least_squares)
# print(least_squares["Squared Difference"].sum())

# game_data = data[["Team 1", "Score 1", "Team 2", "Score 2", "Week", "Year"]]
# game_data.to_csv('game_data.csv', index=False)