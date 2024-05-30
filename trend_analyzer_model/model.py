import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime

from metrics import Metrics

class TrendAnalyzerModel:
    def __init__(self, metrics: list[Metrics] = [], freq: str = '5min', days: int = 14):
        self.days = days
        self.freq = freq

        date_range = pd.date_range(start=datetime.now(), periods=0, freq=self.freq)
        self.df = pd.DataFrame(index=date_range, columns=['cpu_load', 'ram_load', 'net_load'])
        self.model_cpu = SARIMAX(self.df['cpu_load'], order=(1, 1, 1), seasonal_order=(1, 1, 1, self.days)).fit()
        self.model_memory = SARIMAX(self.df['ram_load'], order=(1, 1, 1), seasonal_order=(1, 1, 1, self.days)).fit()
        self.model_network = SARIMAX(self.df['net_load'], order=(1, 1, 1), seasonal_order=(1, 1, 1, self.days)).fit()
        self.update_model(metrics)


    # Функция для дообучения модели SARIMA с новыми данными
    def update_model(self, new_metrics: list[Metrics]):
        start_date = datetime.now()
        date_range = pd.date_range(start=start_date, periods=len(new_metrics), freq=self.freq)
        new_data = [(metric.cpu_load, metric.ram_load, metric.net_load) for metric in new_metrics]
        self.df = pd.concat([self.df, pd.DataFrame(new_data, index=date_range, columns=['cpu_load', 'ram_load', 'net_load'])])
        self.df = self.df.interpolate().resample(self.freq).mean().interpolate().asfreq(self.freq)

        self.df = pd.concat([self.df, new_data])
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

