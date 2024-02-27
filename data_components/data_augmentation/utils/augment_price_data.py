import pywt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from IPython.display import clear_output
from tqdm import tqdm
from pandas_market_calendars import get_calendar

from ta.trend import MACD
from ta.momentum import StochasticOscillator
from tqdm import tqdm
from IPython.display import display


def augment_price_data(df_day: pd.DataFrame, lam: float) -> pd.DataFrame:
    assert all(column in df_day.columns for column in ['open', 'high', 'low', 'close']), "Make sure df_day has columns: ['open', 'high', 'low', 'close', 'time']!"
    num_of_bars_for_day = 79
    num_of_bars_for_week = num_of_bars_for_day * 5
    df_day_synth = pd.DataFrame()
    
    for i in ['open', 'high', 'low', 'close']:
        t = df_day.time
        s = df_day[i]
        if len(s) < 60:
            df_day_synth[i] = s
            continue

        coeffs = pywt.wavedec(s,'db2','sym',level=2)

        err = float('inf')
        while err > 0.3: # gnerate until the error is less than 0.3
            coeffs_with_noise = []
            for coeff in coeffs[1:]:
                coeffs_with_noise.append(coeff + (1.0 - lam)**(len(s)/len(coeff)) * np.random.normal(coeff.mean(), coeff.std(), len(coeff)))
            coeffs_with_noise.insert(0, coeffs[0])
            
            s_r = pywt.waverec(coeffs_with_noise,'db2','sym')
            
            if len(s_r) > len(s):
                s_r = s_r[:-1]
            
            err = sum(abs(s-s_r))/len(s)
            #print("MSE of", i, err)
            #clear_output(wait=False)
            assert err > 10e-4, "The augmentation is just making a copy of the original!"
            #assert err > 0.3, "The augmentation is too much!"
            
            df_day_synth[i] = s_r
    
    return df_day_synth
    
    

# function generates a synthesized data
def obtain_syn_data(original_df:pd.DataFrame, lam:float, start_date:str, end_date:str) -> pd.DataFrame:
    # Get market opening days
    # Get the NYSE calendar
    nyse = get_calendar("XNYS")
    # Get the valid trading days for the specified date range
    trading_days = nyse.valid_days(start_date=start_date, end_date=end_date).date
    
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


def labeling_function(df: pd.DataFrame, label_name: str='label') -> pd.DataFrame:
    """
    Labeling buy, sell, no-action for the given 5-min interval price data(OHLC)
    """
    df.set_index('datetime', inplace=True)
    num_of_bars_for_week = 79 * 5
    df[label_name] = 0  # Initialize the 'label' column with 'no-action'

    for i in tqdm(range(len(df) - num_of_bars_for_week - 79)):
        df_week = df.iloc[i:i+num_of_bars_for_week]
        df_week_after = df.iloc[i+num_of_bars_for_week:i+num_of_bars_for_week+79]
        current_price = df_week['close'].iloc[-1]

        up_target = current_price * (1.0 + 0.0035)  # the upside target is +0.35%
        up_stoploss = current_price * (1.0 - 0.0015)  # the upside stoploss is -0.15%

        down_target = current_price * (1.0 - 0.0035)  # the downside target is -0.35%
        down_stoploss = current_price * (1.0 + 0.0015)  # the downside stoploss is +0.15%

        j = 0
        for index, row in df_week_after.iterrows():
            if row['high'] >= up_target and (df_week_after['low'][:index] >= up_stoploss).all():
                df.loc[df_week.index[-1], label_name] = 1
                break
            elif row['low'] <= down_target and (df_week_after['high'][:index] <= down_stoploss).all():
                df.loc[df_week.index[-1], label_name] = 2
                break
            j += 1
            if j == 79:
                break

    return df




def tech_indicator_calc(spy_df_synth: pd.DataFrame) -> pd.DataFrame:
    # time in second normalized by 09:30:00 == 34200, 16:00:00 == 57600, 57600-34200==23400
    spy_df_synth['time'] = spy_df_synth.index.to_series().apply(lambda x: (x.time().hour * 3600 + x.time().minute * 60 - 34200) / 2340.0)
    
    # add Moving Averages (20 min and 50 min) 
    spy_df_synth['MA50'] = spy_df_synth['close'].rolling(window=50).mean()
    spy_df_synth['MA200'] = spy_df_synth['close'].rolling(window=200).mean()
    
    # MACD
    macd = MACD(close=spy_df_synth['close'], window_slow=26, window_fast=12, window_sign=9) 
    spy_df_synth['macd_diff'] = macd.macd_diff()
    spy_df_synth['macd'] = macd.macd()
    spy_df_synth['macd_signal'] = macd.macd_signal()
    
    # Stochastic
    stoch = StochasticOscillator(high=spy_df_synth['high'], close=spy_df_synth['close'], low=spy_df_synth['low'], 
                                 window=50, smooth_window=10) 
    spy_df_synth['stoch'] = stoch.stoch()
    spy_df_synth['stoch_signal'] = stoch.stoch_signal()

    spy_df_synth.reset_index(inplace=True)
    
    return spy_df_synth
    