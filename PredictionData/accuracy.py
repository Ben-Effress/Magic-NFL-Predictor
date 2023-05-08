import pandas as pd
import matplotlib.pyplot as plt

# Create lists to store week numbers and winner accuracies
week_numbers = []
winner_accuracies = []

# Initialize variables to store total correct and total predictions
total_spread_correct = 0
total_winner_correct = 0
total_predictions = 0

# Loop through weeks 1 to 18
for week in range(1, 19):
    # Read the prediction data from the CSV file for the current week
    filename = f"Week_{week}_predictions.csv"
    data = pd.read_csv(filename)

    # Calculate the number of correct predictions for the current week
    correct_spread_predictions = 0
    correct_winner_predictions = 0
    for _, row in data.iterrows():
        if abs(row["Real Spread"] - row["Spread"]) < 3:
            correct_spread_predictions += 1
        if row["Real Spread"] * row["Spread"] > 0:
            correct_winner_predictions += 1

    # Print the accuracy for the current week
    spread_accuracy = correct_spread_predictions / len(data) * 100
    print(f"Week {week}: {spread_accuracy:.2f}% spread accuracy (within 3)")
    winner_accuracy = correct_winner_predictions / len(data) * 100
    print(f"Week {week}: {winner_accuracy:.2f}% winner accuracy")

    # Update the total correct and total predictions variables
    total_spread_correct += correct_spread_predictions
    total_winner_correct += correct_winner_predictions
    total_predictions += len(data)
    
    # Calculate the winner accuracy for the current week and append to the lists
    winner_accuracy = correct_winner_predictions / len(data) * 100
    week_numbers.append(week)
    winner_accuracies.append(winner_accuracy)

# Print the overall accuracy for all weeks
overall_spread_accuracy = total_spread_correct / total_predictions * 100
print(f"\nOverall spread accuracy: {overall_spread_accuracy:.2f}%")
overall_winner_accuracy = total_winner_correct / total_predictions * 100
print(f"\nOverall winner accuracy: {overall_winner_accuracy:.2f}%")

# Plot the data
plt.plot(week_numbers, winner_accuracies, marker='o')
plt.xlabel('Week')
plt.ylabel('Winner Accuracy (%)')
plt.title('Percent of Correct Winners vs. Week')

# Display the plot
plt.show()
