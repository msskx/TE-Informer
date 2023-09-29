import datetime

import numpy as np
import pandas as pd
# Trigonometric coding of time

def process_data(self, source=r'Air_AQI_yanan.csv'):
    df = pd.read_csv(source)
    timestamps = [ts.split('+')[0] for ts in df['Date']]
    timestamps_day = np.array([float(datetime.datetime.strptime(t, '%Y-%m-%d').day) for t in timestamps])
    timestamps_month = np.array([float(datetime.datetime.strptime(t, '%Y-%m-%d').month) for t in timestamps])
    days_in_month = 30
    month_in_year = 12
    df['sin_day'] = np.sin(2 * np.pi * timestamps_day / days_in_month)
    df['cos_day'] = np.cos(2 * np.pi * timestamps_day / days_in_month)
    df['sin_month'] = np.sin(2 * np.pi * timestamps_month / month_in_year)
    df['cos_month'] = np.cos(2 * np.pi * timestamps_month / month_in_year)
    return df