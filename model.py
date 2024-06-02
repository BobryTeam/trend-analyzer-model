import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime

from metrics import Metrics

class TrendAnalyzerModel:
    def __init__(self, metrics: list[Metrics], freq: str = '5min', days: int = 14):
        self.days = days
        self.freq = freq
        self.df = pd.read_json('[' + ','.join(map(str, metrics)) + ']')
        self.df['date'] = pd.date_range(start=datetime.now(), periods=len(metrics), freq=self.freq)
        self.df.set_index('date', inplace=True)
        self.model_cpu = SARIMAX(self.df['cpu_load'], order=(1, 1, 1), seasonal_order=(1, 1, 1, self.days)).fit()
        self.model_memory = SARIMAX(self.df['ram_load'], order=(1, 1, 1), seasonal_order=(1, 1, 1, self.days)).fit()
        self.model_network = SARIMAX(self.df['net_load'], order=(1, 1, 1), seasonal_order=(1, 1, 1, self.days)).fit()
        self.update_model(metrics)


    # Функция для дообучения модели SARIMA с новыми данными
    def update_model(self, new_metrics: list[Metrics]):
        new_data = pd.read_json('[' + ','.join(map(str, new_metrics)) + ']')
        new_data['date'] = pd.date_range(start=datetime.now(), periods=len(new_metrics), freq=self.freq)
        new_data.set_index('date', inplace=True)
        self.df = pd.concat([self.df, new_data])
        self.df = self.df.interpolate().resample(self.freq).mean().interpolate().asfreq(self.freq)
        self.model_cpu = SARIMAX(self.df['cpu_load'], order=(1, 1, 1), seasonal_order=(1, 1, 1, self.days)).fit()
        self.model_memory = SARIMAX(self.df['ram_load'], order=(1, 1, 1), seasonal_order=(1, 1, 1, self.days)).fit()
        self.model_network = SARIMAX(self.df['net_load'], order=(1, 1, 1), seasonal_order=(1, 1, 1, self.days)).fit()


    # Метод для предсказания CPU загрузки на сервере через 5 минут
    def get_predict(self):
        forecast_cpu = self.model_cpu.forecast(steps=1)
        forecast_memory = self.model_memory.forecast(steps=1)
        forecast_network = self.model_network.forecast(steps=1)

        return forecast_cpu.values[0], forecast_memory.values[0], forecast_network.values[0]

    # Метод для запуска обновления и предсказания модели


    def analyze(self, new_metrics: list[Metrics]):
        self.update_model(new_metrics)
        cpu, ram, net = self.get_predict()
        ram /= self.df['ram_load'].max()
        return cpu, ram, net
