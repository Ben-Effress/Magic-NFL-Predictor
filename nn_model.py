import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from get_game_preview_data import getGamePreviewDataDF

# Define the model
class Net(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 1000, dtype=torch.float64)
        self.fc2 = nn.Linear(1000, 500, dtype=torch.float64)
        self.fc3 = nn.Linear(500, 100, dtype=torch.float64)
        self.fc4 = nn.Linear(100, 1, dtype=torch.float64)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
    def __repr__(self):
        return f'Magic NFL Predictor Net(input_size={self.fc1.in_features}, output_size={self.fc4.out_features})'


def main():
    # Load and preprocess the data
    df = pd.read_csv("game_preview_data.csv")
    # X_home = df.drop(['Spread (Away Score - Home Score)', "Home Team Score", "Away Team Score", "Away Team", "Home Team"], axis=1).values
    # X_away = df.drop(['Spread (Away Score - Home Score)', "Home Team Score", "Away Team Score", "Away Team", "Home Team"], axis=1).values
    X_home = df.drop(['Spread (Away Score - Home Score)', "Home Team Score", "Away Team Score"], axis=1).values
    X_away = df.drop(['Spread (Away Score - Home Score)', "Home Team Score", "Away Team Score"], axis=1).values
    y_home = df['Home Team Score'].values.reshape(-1, 1)
    y_away = df['Away Team Score'].values.reshape(-1, 1)
    spread = df['Spread (Away Score - Home Score)'].values.reshape(-1, 1)

    # Split the data into training and testing sets
    X_home_train, X_home_test, y_home_train, y_home_test, spread_train, spread_test = train_test_split(X_home, y_home, spread, test_size=0.2, random_state=42)
    X_away_train, X_away_test, y_away_train, y_away_test, spread_train, spread_test = train_test_split(X_away, y_away, spread, test_size=0.2, random_state=42)

    # Convert the data to PyTorch tensors
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

    # Create a DataLoader to feed the data to the model in batches
    home_train_dataset = TensorDataset(X_home_train_tensor, y_home_train_tensor)
    home_train_dataloader = DataLoader(home_train_dataset, batch_size=32, shuffle=True)

    away_train_dataset = TensorDataset(X_away_train_tensor, y_away_train_tensor)
    away_train_dataloader = DataLoader(away_train_dataset, batch_size=32, shuffle=True)
    
    
    home_net = Net(184)
    away_net = Net(184)
    
    criterion = nn.L1Loss() # Mean Absolute Error
    optimizer_home = optim.Adam(home_net.parameters(), lr=0.001)
    optimizer_away = optim.Adam(away_net.parameters(), lr=0.001)
    
    epochs = 100
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
        
        if epoch % 77 == 0:
            plt.plot(home_losses, label='Home Team')
            plt.plot(away_losses, label='Away Team')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.legend()
            plt.show()
        
    torch.save(home_net.state_dict(), "home_model.pt")
    torch.save(away_net.state_dict(), "away_model.pt")

if __name__ == '__main__':
    main()
