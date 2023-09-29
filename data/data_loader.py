import datetime
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader


class MYDataset(torch.utils.data.Dataset):
    def __init__(self, is_train=True, source='/data/Air_AQI_yanan.csv',seq_length=30, delay=1):
        df = self.process_data(source=source)
        x, y = self.get_data(df=df, seq_length=seq_length, delay=delay)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        mean = x_train.mean(axis=0)  # 对列处理
        std = x_train.std(axis=0)
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std
        if is_train:
            self.features = x_train
            self.labels = y_train
        else:
            self.features = x_test
            self.labels = y_test

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return len(self.features)

    def get_data(self, df=None, seq_length=30, delay=1):
        data = pd.DataFrame(df.loc[:2146,
                            ['Date', 'AQI', 'PM2_5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'sin_day', 'cos_day',
                             'sin_month',
                             'cos_month']])
        data = data.set_index('Date')
        data.to_csv('./AQI.csv')
        seq_length = seq_length  # 数据观测值
        delay = delay  # 数据预测值
        data_ = []
        for i in range(len(data) - seq_length - delay):
            data_.append(data.iloc[i:i + seq_length + delay])
        data_ = np.array([df.values for df in data_])
        np.random.shuffle(data_)  # 数据打乱
        x = data_[:, :seq_length]  # 高维切片
        y = data_[:, -delay:, 0]  # 取出delay个预测数据，并预测AQI特征
        return x, y

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
