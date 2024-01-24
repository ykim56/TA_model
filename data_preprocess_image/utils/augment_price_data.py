import pywt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from IPython.display import clear_output



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
    