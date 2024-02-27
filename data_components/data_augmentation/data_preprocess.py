import pandas as pd
import os
from pandas_market_calendars import get_calendar
from utils.augment_price_data import augment_price_data
from tqdm import tqdm
import gcsfs

# number of augmented sets
num_of_aug_sets = 2
# Specify the directory path
directory_path = 'gs://ta-charts/raw_data/*.csv'
# augmented data directory
aug_path = 'gs://ta-charts/aug_data'
# List all files in the directory
#all_files = os.listdir(directory_path)

# Filter out files that end with .csv
#csv_files = [file for file in all_files if file.endswith('.csv')]

# Read the CSV file
df = pd.read_csv(directory_path)
df['datetime'] = pd.to_datetime(df['datetime'])

# Get the NYSE calendar
nyse = get_calendar("XNYS")
# Get the valid trading days for the specified date range
trading_days = nyse.valid_days(start_date='2000-01-04', end_date='2023-12-22').date



# function generates a synthesized data
def obtain_syn_data(original_df:pd.DataFrame, lam:float) -> pd.DataFrame:
    synth_frames = []
    for timestamp in tqdm(trading_days):
        original_df_day = original_df[original_df['datetime'].dt.date == timestamp]
        original_df_day_synth = augment_price_data(original_df_day, lam=lam)
        synth_frames.append(original_df_day_synth)

    # Concatenate all DataFrames in the list
    df_synth = pd.concat(synth_frames, axis=0)

    # Optionally reset the index if needed
    df_synth.reset_index(drop=True, inplace=True)
    # Both have the same dates, volume, time
    df_synth[['datetime', 'volume']] = original_df[['datetime', 'volume']]

    print(f"Augmented: {len(df_synth)}")
    print(f"Original: {len(original_df)}")

    return df_synth




for i in tqdm(range(1)):
    spy_df_synth = obtain_syn_data(df, lam=0.5)
    spy_df_synth.to_csv(os.path.join(aug_path, f'aug_{i+1}.csv'), index=False, mode='w')
    