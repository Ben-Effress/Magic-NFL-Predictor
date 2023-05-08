import pandas as pd
import requests
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
import time
import numpy as np
import warnings
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def getGamePreviewDataDF(gid, week_index):
    week_index = int(week_index)
    
    # create a fake user-agent string to emulate a browser
    user_agent = UserAgent().random

    # set the headers to include the user-agent string
    headers = {"User-Agent": user_agent}

    soup = None
    games_div = None
    
    attempts = -1
    
    # Make a print statement about what url is being requested
    print("Requesting gid={}, week_index={}".format(gid, week_index))
    
    while not games_div:
        if attempts > 3: 
            return gid
        attempts += 1
        time.sleep(0.2)
        
        response = requests.get("https://www.footballdb.com/games/preview.html?gid=" + str(gid), headers=headers)
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Check if "Preview is not available for this game" is in the HTML file
        if "Preview is not available for this game" in soup.text:
            return gid
        
        games_div = soup.find('div', {'class': 'rightcol_module'})
        
        if not games_div:
            print("Empty result for gid={}, week_index={}. Trying again...".format(gid, week_index))
        
        
        
        

    # Create an empty list to store the dataframes
    dfs = []

    # Keep track of column names
    col_names = set()
    dup_index = 0

    # print(games_div)

    # create an array of the tables within the 'sbgame_stacked' div
    game_tables = games_div.find_all('table')
    
    with open('my_list.txt', 'w') as file:
        for item in game_tables:
            file.write("%s\n" % item)
    
    tables = games_div.find_all('table')
    
    if (week_index > len(tables) - 1):
        print(f"Week index out of range. Returning {gid}")
        return gid
    else:
        score_table = games_div.find_all('table')[week_index]

    # Get the table headers
    headers = []
    for th in score_table.find_all('th'):
        header_text = th.text.strip()
        if header_text:
            headers.append(header_text)

    # Get the table rows
    rows = []
    for tr in score_table.find_all('tr'):
        row = []
        for td in tr.find_all('td'):
            row.append(td.text.strip())
        if row:
            rows.append(row)
            
    df = pd.DataFrame(rows, columns=headers)

    # Find all tables on the page
    tables = soup.find_all('table')

    # print(df)
    
    # Get just the final score
    if "CANCELLED" in df.columns:
        return gid
    scores = df["T"]
    
    if "FINAL" in headers:
        teams = df["FINAL"]
    else:
        teams = df["F/OT"]
 
    # print(df)

    # Append to dataframes list
    dfs.append(scores)

    # Loop through each table and extract the data into a dataframe
    for table in tables:
        # Get the table headers
        headers = []
        for th in table.find_all('th'):
            header_text = th.text.strip()
            if header_text:
                headers.append(header_text)

        # Get the table rows
        rows = []
        for tr in table.find_all('tr'):
            row = []
            for td in tr.find_all('td'):
                row.append(td.text.strip())
            if len(row) > 1:
                if row[0] in col_names:
                    dup_index += 1
                    row[0] = row[0] + str(dup_index)
                col_names.add(row[0])
            if row:
                rows.append(row)

        # print(rows)
        # print(headers)
        # print(len(headers))
        
        if "FINAL" in headers or "F/OT" in headers:
            break
        if "OFFENSE" not in headers and "DEFENSE" not in headers:
            headers.insert(0, "Category")
            
        if len(headers) > 1:
            # Create the dataframe
            df = pd.DataFrame(rows, columns=headers)
            
            # Transpose the dataframe
            df = df.set_index(headers[0]).T
            # print(df)
            
            # Append to dataframes list
            dfs.append(df)
            
        
            
            
            

    # Concatenate all the dataframes into one
    df = pd.concat(dfs, axis=1)

    # Select the T values and assign them to a new variable
    t_values = df.loc[[0, 1], 'T']

    # Remove the 3rd and 4th rows from the original DataFrame
    df = df.drop([0,1])

    df = df.drop('T', axis=1)

    # Reset the index of the modified DataFrame
    df = df.reset_index(drop=True)

    # Combine the modified DataFrame with the T values DataFrame
    df = df.join(t_values)

    # Rename the column to 'T'
    df = df.rename(columns={'T': 'Score'})

    # print(df)
    if len(df) >= 2:
        df.index = ["Away Team", "Home Team"]
    else:
        return gid

    #factor data for lin reg model = turn to single numerical value

    rename_dict = {
        "Rushing": "Rushing Touchdowns",
        "Passing": "Passing Touchdowns",
        "Returns": "Returns Touchdowns",
        "Rushing1": "Rushing First Downs",
        "Passing2": "Passing First Downs",
        "Penalty": "Penalty First Downs",
        "Touchdowns": "Kickoff Return Touchdowns",
        "Touchdowns3": "Punt Return Touchdowns",
        "Total Touchdowns4": "Opponent Total Touchdowns",
        "Rushing5": "Opponent Rushing Touchdowns",
        "Passing6": "Opponent Passing Touchdowns",
        "Returns7": "Opponent Returns Touchdowns",
        "Field Goals8": "Opponent Field Goals",
        "Extra Points9": "Opponent Extra Points",
        "Rushing10": "Opponent Rushing First Downs",
        "Passing11": "Opponent Passing First Downs",
        "Penalty12": "Opponent Penalty First Downs",
        "Third Down Conversions13": "Opponent Third Down Conversions",
        "Fourth Down Conversions14": "Opponent Fourth Down Conversions",
        "Rushing Yards Per Game15": "Opponent Rushing Yards Per Game",
        "Rushing Attempts16": "Opponent Rushing Attempts",
        "Rushing Average17": "Opponent Rushing Average",
        "Attempts-Completions-Pct18": "Opponent Attempts-Completions-Pct",
        "Average Yards19": "Opponent Average Punt Yards",
        "Average Yards": "Average Punt Yards",
        "Kickoff Return Yards20": "Opponent Kickoff Return Yards",
        "Kickoff Return Average21": "Opponent Kickoff Return Average",
        "Touchdowns22": "Opponent Kickoff Return Touchdowns",
        "Punt Return Yards23": "Opponent Punt Return Yards",
        "Punt Return Average24": "Opponent Punt Return Average",
        "Touchdowns25": "Opponent Punt Return Touchdowns",
        "Intercepted by": "Opponent Intercepted By",
    }

    df = df.rename(columns=rename_dict)

    df["Games Played"] = df["Overall Record"].apply(lambda x: int(x.split('-')[0]) + int(x.split('-')[1]))

    record_columns = ["Overall Record", "Home Record", "Away Record", "Division Record", "Conference Record", "Inter-conference Record"]
    for r_c in record_columns:
        df[r_c] = df[r_c].apply(lambda x: 0 if sum(map(int, x.split('-'))) == 0 else float(int(x.split('-')[0])/(int(x.split('-')[0]) + int(x.split('-')[1])) if int(x.split('-')[0]) + int(x.split('-')[1]) != 0 else 0))
        df = df.rename(columns={r_c: r_c.split()[0] + " Win Percentage"})

    strength_columns = ["Strength of Victory", "Strength of Schedule"]
    for s_c in strength_columns:
        df[s_c] = df[s_c].apply(lambda x: float(x.split("(")[1].split(")")[0]))
        df = df.rename(columns={s_c: s_c + "(Win Percentage)"})
        
    df["Vs. Common Opponents"] = df["Vs. Common Opponents"].apply(lambda x: 0 if sum(map(int, x.split('-'))) == 0 else float(int(x.split('-')[0])/(int(x.split('-')[0]) + int(x.split('-')[1])) if int(x.split('-')[0]) + int(x.split('-')[1]) != 0 else 0))
    df = df.rename(columns={"Vs. Common Opponents": "Vs. Common Opponents Win Percentage"})

    df["Streak"] = df["Streak"].apply(lambda x: int(x[1:]))

    df["Turnover Margin"] = df["Turnover Margin"].astype(int)

    per_game_columns = ["Total Yardage", "Points Scored", "Rushing Yardage", "Passing Yardage", "Total Yardage Allowed", "Points Allowed", "Rushing Yardage Allowed", "Passing Yardage Allowed"]
    for p_c in per_game_columns:
        # print(p_c)
        df[p_c + " Rank"] = df[p_c].apply(lambda x: int(x.split(' ')[0].strip('thrdndst')))

        df[p_c] = df[p_c].apply(lambda x: float(x.split('(')[-1].split(' ')[0]))
        df = df.rename(columns={p_c: p_c + " Per Game"})

    df.drop('TOTAL POINTS SCORED', axis=1, inplace=True)
    df.drop("TOTAL YARDS GAINED", axis=1, inplace=True)
    df.drop("Rushing Yards", axis=1, inplace=True)
    df.drop("Rushing Yards Per Game", axis=1, inplace=True)
    df.drop("Passing Yards (Net)", axis=1, inplace=True)
    df.drop("OPPONENT POINTS SCORED", axis=1, inplace=True)
    df.drop("OPPONENT YARDS GAINED", axis=1, inplace=True)
    df.drop("Opponent Rushing Yards", axis=1, inplace=True)
    df.drop("Opponent Rushing Yards Per Game", axis=1, inplace=True)
    df.drop("Opponent Passing Yards (Net)", axis=1, inplace=True)
        
    df["Times Sacked"] = df["Times Sacked - Yards Lost"].apply(lambda x: int(x.split('-')[0]))
    df["Sack Yards Lost"] = df["Times Sacked - Yards Lost"].apply(lambda x: int(x.split('-')[1]))
    df.drop("Times Sacked - Yards Lost", axis=1, inplace=True)
        
    df["Penalties"] = df["Penalties-Yards"].apply(lambda x: int(x.split('-')[0]))
    df["Penalty Yards"] = df["Penalties-Yards"].apply(lambda x: int(x.split('-')[1]))
    df.drop("Penalties-Yards", axis=1, inplace=True)
        
    df["Fumbles"] = df["Fumbles-Lost"].apply(lambda x: int(x.split('-')[0]))
    df["Fumbles Lost"] = df["Fumbles-Lost"].apply(lambda x: int(x.split('-')[1]))
    df.drop("Fumbles-Lost", axis=1, inplace=True)
        
    df["Sacks"] = df["Sacks - Opponent Yards Lost"].apply(lambda x: int(x.split('-')[0]))
    df["Opponent Sack Yards Lost"] = df["Sacks - Opponent Yards Lost"].apply(lambda x: int(x.split('-')[1]))
    df.drop("Sacks - Opponent Yards Lost", axis=1, inplace=True)
        
    df["Opponent Penalties"] = df["Opponent Penalties-Yards"].apply(lambda x: int(x.split('-')[0]))
    df["Opponent Penalty Yards"] = df["Opponent Penalties-Yards"].apply(lambda x: int(x.split('-')[1]))
    df.drop("Opponent Penalties-Yards", axis=1, inplace=True)
        
    df["Opponent Fumbles"] = df["Opponent Fumbles-Recovered"].apply(lambda x: int(x.split('-')[0]))
    df["Opponent Recovered Fumbles"] = df["Opponent Fumbles-Recovered"].apply(lambda x: int(x.split('-')[1]))
    df.drop("Opponent Fumbles-Recovered", axis=1, inplace=True)

    fraction_columns = ["Field Goals", "Extra Points", "Opponent Field Goals", "Opponent Extra Points"]
    for col in fraction_columns:
        df[col] = df[col].apply(lambda x: 0 if int(x.split('/')[1]) == 0 else int(x.split('/')[0])/float(x.split('/')[1]))
        df = df.rename(columns={col: col + " Percentage"})

    percent_columns = ["Third Down Conversions", "Fourth Down Conversions", "Attempts-Completions-Pct", "Opponent Third Down Conversions", "Opponent Fourth Down Conversions", "Opponent Attempts-Completions-Pct"]
    for col in percent_columns:
        # get the percent value from a string like 5-14 (35.7%)
        df[col] = df[col].apply(lambda x: float(x.split('(')[-1].split('%')[0]))
        df = df.rename(columns={col: col + " Percentage"})

    #df.to_csv("data.csv")
    away_team_df = df.iloc[0:1, :]
    home_team_df = df.iloc[1:2, 0:]

    away_team_df.columns = ["Away Team " + col for col in away_team_df.columns]
    home_team_df.columns = ["Home Team " + col for col in home_team_df.columns]

    pd.options.mode.chained_assignment = None

    dfs = [("Away Team ", away_team_df), ("Home Team ", home_team_df)]

    for pre, dataFrame in dfs:                           
        to_per_game_cols = ["Total Touchdowns", "Rushing Touchdowns", "Passing Touchdowns", "Returns Touchdowns", "FIRST DOWNS", "Rushing First Downs", "Passing First Downs", "Penalty First Downs", "Rushing Attempts", "Times Sacked", "Sack Yards Lost", "Had Intercepted", "Punts", "Kickoff Returns", "Kickoff Return Yards", "Kickoff Return Touchdowns", "Punt Returns", "Punt Return Yards", "Punt Return Touchdowns", "Penalties", "Penalty Yards", "Fumbles", "Fumbles Lost", "Opponent Total Touchdowns", "Opponent Rushing Touchdowns", "Opponent Passing Touchdowns", "Opponent Returns Touchdowns", "OPPONENT FIRST DOWNS", "Opponent Rushing First Downs", "Opponent Passing First Downs", "Opponent Penalty First Downs", "Opponent Rushing Attempts", "Sacks", "Opponent Sack Yards Lost", "Opponent Intercepted By", "Opponent Punts", "Opponent Kickoff Returns", "Opponent Kickoff Return Yards", "Opponent Kickoff Return Touchdowns", "Opponent Punt Returns", "Opponent Punt Return Yards", "Opponent Punt Return Touchdowns", "Opponent Penalties", "Opponent Penalty Yards", "Opponent Fumbles", "Opponent Recovered Fumbles"]
        
        to_per_game_cols = [pre + col for col in to_per_game_cols]
        # print(to_per_game_cols)
        
        for col in to_per_game_cols:
            # print(col)
            new_col_name = col + " Per Game"
            if dataFrame[col].dtype == "object":
                dataFrame[col] = dataFrame[col].str.replace(',', '').astype(float)
            dataFrame[new_col_name] = np.divide(dataFrame[col], dataFrame[pre + "Games Played"], out=np.zeros_like(dataFrame[col]), where=dataFrame[pre + "Games Played"]!=0)
            
        dataFrame.drop(to_per_game_cols, axis=1, inplace=True)
        # select columns of object type and convert them to float
        dataFrame[dataFrame.select_dtypes(include=['object']).columns] = dataFrame.select_dtypes(include=['object']).apply(pd.to_numeric)
        

    # print(away_team_df)
    # print(home_team_df)

    # print(away_team_df.info())
    # print(home_team_df.info())
    home_team_df.to_csv("home_team.csv")

    #reset indexes of home and away dfs
    home_team_df.reset_index(drop=True, inplace=True)
    away_team_df.reset_index(drop=True, inplace=True)

    combined_df = pd.concat([home_team_df, away_team_df], axis=1)
    combined_df = pd.concat([combined_df, away_team_df["Away Team Score"] - home_team_df["Home Team Score"]], axis=1)
    combined_df.columns = list(combined_df.columns[:-1]) + ["Spread (Away Score - Home Score)"]
    # print("Teams:", teams[0])
    combined_df["Away Team"] = teams[0]
    combined_df["Home Team"] = teams[1]

    combined_df = combined_df.copy() # create a copy to de-fragment the DataFrame


    combined_df.to_csv("combined.csv")
    
    return combined_df