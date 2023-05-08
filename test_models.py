import torch
from nn_model import Net
from get_game_preview_data import getGamePreviewDataDF
from sklearn.preprocessing import StandardScaler
from extensions import interpret_json
import pandas as pd
import time

all_predictions = []
failed_exts = []

# Loop through all weeks
for week in range(1, 19):
    extensions = interpret_json("2022.json", week)

    week_dfs = []
    for (ext, week_index) in extensions:
        game_df = getGamePreviewDataDF(ext, week_index)
        retry_count = 0
        
        while type(game_df) == str and retry_count < 3:  # Try up to 3 times
            time.sleep(1)  # Wait for 1 second before retrying
            game_df = getGamePreviewDataDF(ext, week_index)
            retry_count += 1

        if type(game_df) == str:
            failed_exts.append((week, ext))
            failed_exts_df = pd.DataFrame(failed_exts, columns=['Week', 'Failed Extension'])
            failed_exts_df.to_csv('failed_extensions.csv', header=False, index=False)
            print("Failed to retrieve data for game", ext)


        if type(game_df) != str:
            # print(game_df.columns)
            home_team = game_df["Home Team"].values[0]
            away_team = game_df["Away Team"].values[0]
            home_team_score = game_df["Home Team Score"]
            away_team_score = game_df["Away Team Score"]
            game_df.drop("Home Team Score", axis=1, inplace=True)
            game_df.drop("Away Team Score", axis=1, inplace=True)
            game_df.drop(["Spread (Away Score - Home Score)", "Away Team", "Home Team"], axis=1, inplace=True)

            # Convert the data to PyTorch tensors
            game_tensor = torch.tensor(game_df.values, dtype=torch.float64)

            # Load the home score model
            home_model = Net(input_size=game_df.shape[1])
            home_model.load_state_dict(torch.load('models/home_model.pt'))
            home_model.eval()

            # Load the away score model
            away_model = Net(input_size=game_df.shape[1])
            away_model.load_state_dict(torch.load('models/away_model.pt'))
            away_model.eval()

            # Make predictions for the home and away scores
            with torch.no_grad():
                home_model.eval()
                away_model.eval()
                y_pred_home = home_model(game_tensor)
                y_pred_away = away_model(game_tensor)

            # Convert the PyTorch tensors to NumPy arrays and round the predictions to integers
            y_pred_home = y_pred_home.numpy()
            y_pred_away = y_pred_away.numpy()

            # Calculate the spread
            spread_pred = y_pred_away - y_pred_home

            # Print the predicted scores and spread
            print("Predicted scores:")
            print(f"{home_team}: {y_pred_home[0][0]}")
            print(f"{away_team}: {y_pred_away[0][0]}")
            print(f"Spread: {spread_pred[0][0]}")
            print("Real {}: {}".format(home_team, home_team_score.values[0]))
            print("Real {}: {}".format(away_team, away_team_score.values[0]))
            print("Real Spread: {}".format(away_team_score.values[0] - home_team_score.values[0]))
            # print(home_model)

            # Define the data as a list of dictionaries
            data = [
                {'Home Team': home_team, 'Away Team': away_team, 'Home': y_pred_home[0][0], 'Away': y_pred_away[0][0], 'Spread': spread_pred[0][0], 'Real Home': home_team_score.values[0], 'Real Away': away_team_score.values[0], 'Real Spread': away_team_score.values[0] - home_team_score.values[0]},
            ]

            # Create a DataFrame from the data
            df = pd.DataFrame(data)
            
            # Append to the rest of the week
            week_dfs.append(df)
    
    week_df = pd.concat(week_dfs)
    
    # Save to a CSV file
    week_df.to_csv(f"PredictionData/Week_{week}_predictions.csv")
    
    # Append the DataFrame to an existing DataFrame containing other predictions
    all_predictions.append(df)
        
all_predictions = pd.concat(all_predictions)

all_predictions.to_csv("2022_Regular_Season_Predictions.csv")

