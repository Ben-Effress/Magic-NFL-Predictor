from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('game_preview_data.csv')
df.drop("Home Team Score", axis=1, inplace=True)
df.drop("Away Team Score", axis=1, inplace=True)

# Split the data into features and target variable
X = df.drop('Spread (Away Score - Home Score)', axis=1)
y = df['Spread (Away Score - Home Score)']

# Perform PCA on the features
pca = PCA(n_components=20)
X_pca = pca.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Create a Random Forest Regression model with 100 trees
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model to the training data
rf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = rf.predict(X_test)

# Calculate the mean squared error of the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error: ", mse)

# Plot the actual vs predicted spread
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot(y_test, y_test, color='red')
plt.xlabel('Actual Spread')
plt.ylabel('Predicted Spread')
plt.show()

# Get feature importances
importances = rf.feature_importances_

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Plot the feature importances
plt.figure(figsize=(10,6))
plt.title("Feature importances")
plt.bar(range(X_pca.shape[1]), importances[indices])
plt.xticks(range(X_pca.shape[1]), indices, rotation=90)
plt.tight_layout()
plt.show()

# Make predictions on new data using the trained model
X_new = X.iloc[0:10]
X_new_pca = pca.transform(X_new)
y_new_pred = rf.predict(X_new_pca)
print("Predicted spreads for new data: ", y_new_pred)
