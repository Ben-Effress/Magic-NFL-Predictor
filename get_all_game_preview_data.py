import get_game_preview_data as gd
from extensions import get_game_extensions

failed_exts = []
super_failed_exts = []
all_extensions = []
api_key = '81c23e67542b43cab5a2495d6e20c9ec'
for year in range(2013, 2023):
    extensions = get_game_extensions(api_key, year)
    print(extensions)
    all_extensions.extend(extensions)
all_extensions.extend(['2022101002', '2022121813', '2022121814', '2022121810', '2023010801'])

for ext, week_index in all_extensions:
    new_df = gd.getGamePreviewDataDF(ext, week_index)
    if type(new_df) is not str:
        # Append the DataFrame to the end of the CSV file
        with open('game_preview_data.csv', 'a') as f:
            new_df.to_csv(f, mode='a', header=False, index=False)
    else:
        failed_exts.append(ext)
        print("Failed to retrieve data for game", ext)

# Attempt to retrieve data for failed GIDs
for ext in failed_exts:
    new_df = gd.getGamePreviewDataDF(ext)
    
    # Check if data was successfully retrieved
    if type(new_df) is not str:
        # Append the DataFrame to the end of the CSV file
        with open('game_preview_data.csv', 'a') as f:
            new_df.to_csv(f, mode='a', header=False, index=False)
    else:
        super_failed_exts.append(ext)
        print("Failed to retrieve data for game", ext)

print(super_failed_exts)
