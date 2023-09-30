import datetime
import torch
import torch.nn as nn
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

class TrigonometricEncoding(nn.Module):
    def __init__(self, source=r'Air_AQI_yanan.csv'):
        super(TrigonometricEncoding, self).__init__()
        self.source = source

    def forward(self, x):
        df = pd.read_csv(self.source)
        timestamps = [ts.split('+')[0] for ts in df['Date']]
        timestamps_day = np.array([float(datetime.datetime.strptime(t, '%Y-%m-%d').day) for t in timestamps])
        timestamps_month = np.array([float(datetime.datetime.strptime(t, '%Y-%m-%d').month) for t in timestamps])
        days_in_month = 30
        month_in_year = 12
        df['sin_day'] = np.sin(2 * np.pi * timestamps_day / days_in_month)
        df['cos_day'] = np.cos(2 * np.pi * timestamps_day / days_in_month)
        df['sin_month'] = np.sin(2 * np.pi * timestamps_month / month_in_year)
        df['cos_month'] = np.cos(2 * np.pi * timestamps_month / month_in_year)
        # Assuming x is a placeholder for your input data (you need to replace this with your actual input)
        # You can return the processed data as a tensor
        return torch.tensor(df.values, dtype=torch.float32)

# Usage
source_file = 'Air_AQI_yanan.csv'
data_processor = TrigonometricEncoding(source=source_file)
input_data = torch.randn(10, 10) 
output_data = data_processor(input_data)
