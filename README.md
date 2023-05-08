## Video Walk-Through: 
https://drive.google.com/file/d/1XOoPx5RVSSvnSUeQN4Mn_2SWbteYciAM/view?usp=share_link

## Magic NFL Predictor
Magic NFL Predictor is a Python program that uses deep learning to predict the scores of NFL games. The program trains two separate neural networks, one for the home team and one for the away team, and uses these models to predict the scores for a given game. The program takes game data from https://www.footballdb.com/games/previews.html.

## Dependencies
To run this program, you will need the following dependencies:

torch
pandas
scikit-learn
matplotlib
tqdm
requests
fake_useragent
bs4
You can install these dependencies via pip:
pip install torch pandas scikit-learn matplotlib tqdm requests fake_useragent bs4

## Usage
Run the main program:
python3 Final_Project.py
When prompted, paste the gid (game ID) from the game that you want to predict from https://www.footballdb.com/games/previews.html (from the URL).
Enter the week index when prompted. You can find this by going to the preview page and looking at the scoreboard. Type the position of the game in the array of games (zero-indexed).
The program will predict the scores for the home and away teams and display the results.

## Program Structure
The program consists of several functions that perform various tasks:
load_and_preprocess_data: Load and preprocess data from the CSV file.
create_train_test_splits: Create train and test splits for the dataset.
create_tensors: Create tensors from the train and test data.
create_dataloaders: Create DataLoaders for home and away team training data.
train_models: Train home and away team models.
save_models: Save trained models to files.
get_models: Train and save models for home and away teams, and display the training losses.
create_single_tensors: Create single tensors for home and away team data.
load_models: Load trained models from files.
predict_scores: Predict home and away team scores using trained models.
predict_game_score: Predict game scores based on user input and display the results.

## Customizing the Model
To change the model's architecture or training parameters, you can modify the Net class definition and the train_models function in the main program. You can also adjust the batch size and the number of training epochs as needed.

## Website
I made the Magic NFL Predictor into a website that shows the predictions and real scores for NFL games from the 2022-2023 regular season.
The website can be found at https://test-vshkicb2dq-uc.a.run.app/. I used flask, docker, and Google Cloud Run to deploy the website.

## Data Collection
The game_preview_data.csv file game preview data was collected using BeautifulSoup.

## Notes
I also tried some other models like a LinearRegression model and a RandomForests model, but none were as good as the neural network model.
I also tried an elo system that was unfinished because I was more interested in predicted the spread and scores than just the winner.

## Future plans
I plan to refine the ELO system to get a working win probability system. Additionally, I will add the team elos as a feature and then
retrain the neural network. I may also include betting odds and sportsbook over/under and spread lines as features. Funnily, this bring me to my final future plan: chasing +EV. I plan to use this revamped model to revamp the website as a betting model. Combining the neural
network, elo system, and true betting lines, I want to show the EV for bets based on the models, true odds, etc. Also, the website is
very ugly, so redesigning that is also a future plan.
