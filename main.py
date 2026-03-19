import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

data_file_path = "./data/data.xlsx"
sheets = list(map(str, range(2022, 2027)))

row = 8  # строка
start_col = 1  # начальный столбец

# Собираем данные в один список
frame = []
for sheet in sheets:
    frame += pd.read_excel(data_file_path, sheet).iloc[row, start_col:].to_list()


# Очистка данных
df = pd.DataFrame(frame)
z_scores = np.abs((df - df.mean()) / df.std())
df = df[z_scores < 3]
# Восстанавливаем пропущенные значения по среднему соседних
df.fillna(df.mean(), inplace=True)


# Обучение модели
model = ARIMA(df, order=(1, 1, 1))
model_fit = model.fit()

# Прогноз на основе обученной модели
steps = 30
forecast = model_fit.forecast(steps=steps)

# Расчет СКО и средней абсолютной ошибки
mse = mean_squared_error(df[-steps:], forecast)
mae = mean_absolute_error(df[-steps:], forecast)

print(mse)
print(mae)
#
# Визуализация исходных данных и прогноза и
# сохранение в файл для отображения на странице
plt.plot(df.index[:-steps], df[:-steps], label="Исходные данные")
plt.plot(df.index[-steps:], df[-steps:], label="Прогноз")
plt.title("Статистика и прогноз на 6 месяцев")
plt.xlabel("Месяцы")
plt.ylabel("Индекс")
plt.legend()
plt.grid(True)
plt.show()
