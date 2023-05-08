import requests, json

def get_game_extensions(api_key, season):
    url = f'https://api.sportsdata.io/v3/nfl/scores/json/Schedules/{season}?key={api_key}'
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception('Failed to retrieve game schedules')
    schedules = response.json()
    extensions = []
    curr_date = ""
    index = 1
    week_index = 0
    curr_week = 0

    for game in schedules:
        if game["AwayTeam"] != "BYE" and game["HomeTeam"] != "BYE":
            if (date := game["Date"]) != curr_date:
                index = 1
                curr_date = date
                if game["Week"] != curr_week:
                    week_index = 0
                    curr_week = game["Week"]
            else:
                index += 1
            day_index = "0" + str(index) if index < 10 else str(index)
            ext = date[:4] + date[5:7] + date[8:10] + str(week_index)
            extensions.append((ext, week_index))
            week_index += 1
    return extensions


def interpret_json(json_file, week):

    # Open the JSON file for reading
    with open(json_file, 'r') as file:
        # Load the JSON data from the file
        data = json.load(file)

    # Access the data as a list
    print(data)
    extensions = []
    curr_date = ""
    index = 1
    week_index = 0
    
    for game in data:
        if game["AwayTeam"] != "BYE" and game["HomeTeam"] != "BYE":
            if (date := game["Date"]) != curr_date:
                index = 1
                curr_date = date
            else:
                index += 1
            
            day_index = "0" + str(index) if index < 10 else str(index)
            ext = date[:4] + date[5:7] + date[8:10] + day_index
            if game["Week"] == week:
                extensions.append((ext, week_index))
                week_index += 1
            elif game["Week"] > week:
                break
    return extensions
