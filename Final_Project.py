import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from get_game_preview_data import getGamePreviewDataDF

# Define the model
class Net(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 100, dtype=torch.float64)
        self.fc2 = nn.Linear(100, 50, dtype=torch.float64)
        self.fc3 = nn.Linear(50, 100, dtype=torch.float64)
        self.fc4 = nn.Linear(100, 1, dtype=torch.float64)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
    def __repr__(self):
        return f'Magic NFL Predictor Net(input_size={self.fc1.in_features}, output_size={self.fc4.out_features})'

def load_and_preprocess_data(filepath):
    """Load and preprocess data from the CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        tuple: A tuple containing preprocessed X_home, X_away, y_home, y_away, and spread data.
    """
    df = pd.read_csv(filepath)
    X_home = df.drop(['Spread (Away Score - Home Score)', "Home Team Score", "Away Team Score"], axis=1).values
    X_away = df.drop(['Spread (Away Score - Home Score)', "Home Team Score", "Away Team Score"], axis=1).values
    y_home = df['Home Team Score'].values.reshape(-1, 1)
    y_away = df['Away Team Score'].values.reshape(-1, 1)
    spread = df['Spread (Away Score - Home Score)'].values.reshape(-1, 1)
    return X_home, X_away, y_home, y_away, spread

def create_train_test_splits(X_home, X_away, y_home, y_away, spread, test_size=0.2, random_state=42):
    """Create train and test splits for the dataset.

    Args:
        X_home (array): Home team feature data.
        X_away (array): Away team feature data.
        y_home (array): Home team target data.
        y_away (array): Away team target data.
        spread (array): Spread data.
        test_size (float, optional): Proportion of dataset to include in the test split. Defaults to 0.2.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple: A tuple containing train and test splits for home and away team data, and spread data.
    """
    X_home_train, X_home_test, y_home_train, y_home_test, spread_train, spread_test = train_test_split(X_home, y_home, spread, test_size=test_size, random_state=random_state)
    X_away_train, X_away_test, y_away_train, y_away_test, spread_train, spread_test = train_test_split(X_away, y_away, spread, test_size=test_size, random_state=random_state)
    return X_home_train, X_home_test, y_home_train, y_home_test, X_away_train, X_away_test, y_away_train, y_away_test, spread_train, spread_test

def create_tensors(X_home_train, X_home_test, y_home_train, y_home_test, X_away_train, X_away_test, y_away_train, y_away_test, spread_train, spread_test):
    """Create tensors from the train and test data.

    Args:
        X_home_train (array): Home team training feature data.
        X_home_test (array): Home team testing feature data.
        ...
        spread_test (array): Spread testing data.

    Returns:
        tuple: A tuple containing tensors for home and away team train and test data, and spread data.
    """
    X_home_train_tensor = torch.tensor(X_home_train, dtype=torch.float64)
    y_home_train_tensor = torch.tensor(y_home_train, dtype=torch.float64)
    X_home_test_tensor = torch.tensor(X_home_test, dtype=torch.float64)
    y_home_test_tensor = torch.tensor(y_home_test, dtype=torch.float64)
    spread_train_tensor = torch.tensor(spread_train, dtype=torch.float64)
    spread_test_tensor = torch.tensor(spread_test, dtype=torch.float64)

    X_away_train_tensor = torch.tensor(X_away_train, dtype=torch.float64)
    y_away_train_tensor = torch.tensor(y_away_train, dtype=torch.float64)
    X_away_test_tensor = torch.tensor(X_away_test, dtype=torch.float64)
    y_away_test_tensor = torch.tensor(y_away_test, dtype=torch.float64)

    return X_home_train_tensor, y_home_train_tensor, X_home_test_tensor, y_home_test_tensor, spread_train_tensor, spread_test_tensor, X_away_train_tensor, y_away_train_tensor, X_away_test_tensor, y_away_test_tensor

def create_dataloaders(X_home_train_tensor, y_home_train_tensor, X_away_train_tensor, y_away_train_tensor, batch_size=32):
    """Create DataLoaders for home and away team training data.

    Args:
        X_home_train_tensor (torch.Tensor): Home team training feature tensor.
        y_home_train_tensor (torch.Tensor): Home team training target tensor.
        X_away_train_tensor (torch.Tensor): Away team training feature tensor.
        y_away_train_tensor (torch.Tensor): Away team training target tensor.
        batch_size (int, optional): Batch size for the DataLoader. Defaults to 32.

    Returns:
        tuple: A tuple containing DataLoaders for home and away team training data.
    """
    home_train_dataset = TensorDataset(X_home_train_tensor, y_home_train_tensor)
    home_train_dataloader = DataLoader(home_train_dataset, batch_size=batch_size, shuffle=True)

    away_train_dataset = TensorDataset(X_away_train_tensor, y_away_train_tensor)
    away_train_dataloader = DataLoader(away_train_dataset, batch_size=batch_size, shuffle=True)

    return home_train_dataloader, away_train_dataloader

def train_models(home_train_dataloader, away_train_dataloader, input_size, epochs=100):
    """Train home and away team models.

    Args:
        home_train_dataloader (DataLoader): DataLoader for home team training data.
        away_train_dataloader (DataLoader): DataLoader for away team training data.
        input_size (int): Input size for the models.
        epochs (int, optional): Number of training epochs. Defaults to 100.

    Returns:
        tuple: A tuple containing trained home and away team models, and their respective training losses.
    """
    home_net = Net(input_size)
    away_net = Net(input_size)

    criterion = nn.L1Loss()  # Mean Absolute Error
    optimizer_home = optim.Adam(home_net.parameters(), lr=0.001)
    optimizer_away = optim.Adam(away_net.parameters(), lr=0.001)

    home_losses = []
    away_losses = []

    for epoch in tqdm(range(epochs)):
        # Train the home team model
        home_net.train()
        train_loss_home = 0.0
        for i, data in enumerate(home_train_dataloader):
            inputs, labels = data
            optimizer_home.zero_grad()
            outputs = home_net(inputs)
            loss_home = criterion(outputs, labels)
            loss_home.backward()
            optimizer_home.step()
            train_loss_home += loss_home.item()
        home_losses.append(train_loss_home / len(home_train_dataloader))

        # Train the away team model
        away_net.train()
        train_loss_away = 0.0
        for i, data in enumerate(away_train_dataloader):
            inputs, labels = data
            optimizer_away.zero_grad()
            outputs = away_net(inputs)
            loss_away = criterion(outputs, labels)
            loss_away.backward()
            optimizer_away.step()
            train_loss_away += loss_away.item()

        away_losses.append(train_loss_away / len(away_train_dataloader))

    return home_net, away_net, home_losses, away_losses

def save_models(home_net, away_net, home_model_path="home_model.pt", away_model_path="away_model.pt"):
    """Save trained models to files.

    Args:
        home_net (Net): Trained home team model.
        away_net (Net): Trained away team model.
        home_model_path (str, optional): File path for the home team model. Defaults to "home_model.pt".
        away_model_path (str, optional): File path for the away team model. Defaults to "away_model.pt".
    """
    torch.save(home_net.state_dict(), home_model_path)
    torch.save(away_net.state_dict(), away_model_path)

def get_models():
    """Train and save models for home and away teams, and display the training losses."""
    X_home, X_away, y_home, y_away, spread = load_and_preprocess_data("game_preview_data.csv")
    X_home_train, X_home_test, y_home_train, y_home_test, X_away_train, X_away_test, y_away_train, y_away_test, spread_train, spread_test = create_train_test_splits(X_home, X_away, y_home, y_away, spread)
    X_home_train_tensor, y_home_train_tensor, X_home_test_tensor, y_home_test_tensor, spread_train_tensor, spread_test_tensor, X_away_train_tensor, y_away_train_tensor, X_away_test_tensor, y_away_test_tensor = create_tensors(X_home_train, X_home_test, y_home_train, y_home_test, X_away_train, X_away_test, y_away_train, y_away_test, spread_train, spread_test)
    home_train_dataloader, away_train_dataloader = create_dataloaders(X_home_train_tensor, y_home_train_tensor, X_away_train_tensor, y_away_train_tensor)

    input_size = X_home_train_tensor.shape[1]
    home_net, away_net, home_losses, away_losses = train_models(home_train_dataloader, away_train_dataloader, input_size)

    save_models(home_net, away_net)

    plt.plot(home_losses, label='Home Team')
    plt.plot(away_losses, label='Away Team')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()
    
def create_single_tensors(X_home, X_away):
    """Create single tensors for home and away team data.

    Args:
        X_home (array): Home team feature data.
        X_away (array): Away team feature data.

    Returns:
        tuple: A tuple containing single tensors for home and away team data.
    """
    X_home_tensor = torch.tensor(X_home, dtype=torch.float64).unsqueeze(0)
    X_away_tensor = torch.tensor(X_away, dtype=torch.float64).unsqueeze(0)

    return X_home_tensor, X_away_tensor

def load_models(home_model_path, away_model_path):
    """Load trained models from files.

    Args:
        home_model_path (str): File path for the home team model.
        away_model_path (str): File path for the away team model.

    Returns:
        tuple: A tuple containing loaded home and away team models.
    """
    home_net = Net(184)
    home_net.load_state_dict(torch.load(home_model_path))
    home_net.eval()

    away_net = Net(184)
    away_net.load_state_dict(torch.load(away_model_path))
    away_net.eval()

    return home_net, away_net

def predict_scores(home_net, away_net, df):
    """Predict home and away team scores using trained models.

    Args:
        home_net (Net): Trained home team model.
        away_net (Net): Trained away team model.
        df (pandas.DataFrame): DataFrame containing game preview data.

    Returns:
        tuple: A tuple containing predicted home and away team scores.
    """
    X_home = df.drop(['Spread (Away Score - Home Score)', "Home Team Score", "Away Team Score", "Away Team", "Home Team"], axis=1).values
    X_away = df.drop(['Spread (Away Score - Home Score)', "Home Team Score", "Away Team Score", "Away Team", "Home Team"], axis=1).values

    X_home_tensor, X_away_tensor = create_single_tensors(X_home, X_away)

    with torch.no_grad():
        home_scores = home_net(X_home_tensor).numpy()
        away_scores = away_net(X_away_tensor).numpy()

    return home_scores, away_scores


def predict_game_score():
    """Predict game scores based on user input and display the results."""
    gid = input("Paste the gid from the game that you want from https://www.footballdb.com/games/previews.html: ")
    week_index = input("Week index (find this by going to the preview page and looking at the scoreboard. Type the position in that array of games): ")

    df = getGamePreviewDataDF(gid, week_index)
    attempts = 0

    while type(df) == str:
        attempts += 1
        print(f"Something went wrong. Try again. Attempts: {attempts}")
        gid = input("Paste the gid from the game that you want from https://www.footballdb.com/games/previews.html: ")
        week_index = input("Week index (find this by going to the preview page and looking at the scoreboard. Type the position in that array of games): ")
        df = getGamePreviewDataDF(gid, week_index)
        
    home_team = df["Home Team"]
    away_team = df["Away Team"]

    home_net, away_net = load_models("home_model.pt", "away_model.pt")
    home_scores, away_scores = predict_scores(home_net, away_net, df)

    print(f"Predicted {home_team.values[0]} Score: {float(home_scores[0][0]):.2f}")
    print(f"Predicted {away_team.values[0]} Score: {float(away_scores[0][0]):.2f}")

if __name__ == '__main__':
    if "home_model.pt" not in os.listdir(".") and "away_model.pt" not in os.listdir("."): 
        get_models()
        predict_game_score()
    else:
        predict_game_score()


