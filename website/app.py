from flask import Flask, render_template
import csv

app = Flask(__name__)

def read_csv(file_path):
    games = []
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            row[0] = int(row[0])  # Convert the game number to an integer
            row[3:] = [float(x) for x in row[3:]]  # Convert numeric values to floats
            games.append(row)
    return games

@app.route('/')
def home():
    games = read_csv("PredictionData/Week_1_predictions.csv")
    return render_template("game_data.html", games=games)

@app.route('/week<int:week>')
def show_week_data(week):
    file_path = f"PredictionData/Week_{week}_predictions.csv"
    games = read_csv(file_path)
    return render_template("game_data.html", games=games)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--port', type=int, default=8000, help='Port number to use for Flask app')
    # args = parser.parse_args()
    app.run(debug=True, port=8000)
