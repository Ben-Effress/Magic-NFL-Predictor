import pandas as pd

# read the data from the CSV file
df = pd.read_csv('real_qb_adjusted_elo_ratings.csv')

# sort the data frame by ELO scores in descending order
ranked_df = df.sort_values(by=['ELO'], ascending=False)

# display the ranked data frame
print(ranked_df)

ranked_df.to_csv('ranked_qb_adjusted_elo_ratings.csv', index=False)
