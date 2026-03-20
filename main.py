import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA


def read_data(row: int, start_col: int) -> list[float]:
    # Собираем данные в один список
    frame = []
    for sheet in sheets:
        frame += pd.read_excel(data_file_path, sheet).iloc[row, start_col:].to_list()

    # print(frame)
    return frame


def clear_data(data: list[float]):
    # Очистка данных
    df = pd.DataFrame(data)
    z_scores = np.abs((df - df.mean()) / df.std())
    df = df[z_scores < 3]
    # Восстанавливаем пропущенные значения по среднему соседних
    df.fillna(df.mean(), inplace=True)

    # Проверка данных на нормальное распределение по Харке—Бера
    jb_test = sm.stats.stattools.jarque_bera(df)
    print("jb_test", jb_test)
    return df


def forecast(data_frame, steps: int):
    # Обучение модели
    model = ARIMA(data_frame, order=(1, 1, 1))
    model_fit = model.fit()

    # Прогноз на основе обученной модели
    forecast = model_fit.forecast(steps=steps)

    # Расчет СКО и средней абсолютной ошибки
    mse = mean_squared_error(data_frame[-steps:], forecast)
    mae = mean_absolute_error(data_frame[-steps:], forecast)
    print("mse", mse, "mae", mae)

    # Создаем DataFrame для прогноза
    forecast_result = model_fit.get_forecast(steps=steps)
    # Создаем индекс для прогноза (продолжение текущего индекса)
    last_index: int = data_frame.index[-1]  # type: ignore
    forecast_index = range(last_index + 1, last_index + steps + 1)  # 214, 215, ... 243

    # Формируем DataFrame прогноза
    forecast_df = pd.DataFrame(
        {
            0: forecast_result.predicted_mean.values  # Сохраняем имя колонки как 0
        },
        index=forecast_index,
    )
    full_df = pd.concat([data_frame, forecast_df])
    return full_df


data_file_path = "./data/data.xlsx"
sheets = list(map(str, range(2022, 2027)))

# Колбаса полукапченая, вареная
row = 8  # строка
start_col = 1  # начальный столбец
steps = 6

raw_data = read_data(row, start_col)
df = clear_data(raw_data)
df = forecast(df, steps)
# gasoline
row = 78
start_col = 1
raw_data = read_data(row, start_col)
gasoline_df = clear_data(raw_data)
gasoline_df = forecast(gasoline_df, steps)


# Визуализация исходных данных и прогноза и
# сохранение в файл для отображения на странице
plt.plot(df.index[:-steps], df[:-steps], label="Колбаса")
plt.plot(df.index[-steps:], df[-steps:], label="Колбаса предсказание")
plt.plot(gasoline_df.index[:-steps], gasoline_df[:-steps], label="Бензин")
plt.plot(gasoline_df.index[-steps:], gasoline_df[-steps:], label="Бензин предсказание")
plt.title("Статистика и прогноз на 6 месяцев")
plt.legend()
plt.grid(True)
plt.show()
