import pandas as pd
import os
from pandas_market_calendars import get_calendar
from utils.augment_price_data import augment_price_data
from tqdm import tqdm
import gcsfs
from google.cloud import storage
from io import StringIO


project_id ='ta-model-data-preprocess'
# number of augmented sets
num_of_aug_sets = 2
# Specify the directory path
directory_path = 'gs://ta-charts-data/origin_data/*.csv'
# augmented data directory
aug_path = 'augmented_data'

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


# Initialize GCS Client
client = storage.Client()
bucket_name = 'ta-charts-data'
bucket = client.bucket(bucket_name)

for i in tqdm(range(num_of_aug_sets)):
    spy_df_synth = obtain_syn_data(df, lam=0.5)
    # Convert DataFrame to CSV String
    spy_df_synth = spy_df_synth.to_csv(index=False)
    # Upload CSV to GCS
    blob = bucket.blob(os.path.join(aug_path, f'aug_{i+1}', f'aug_{i+1}.csv'))
    blob.upload_from_string(spy_df_synth, content_type='text/csv')
    #blob.event_based_hold = True  # Place an event-based hold on the object
    #blob.patch()
    