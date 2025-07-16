import pandas as pd
import numpy as np

# Загружаем файл наоборот
with open('file.csv', encoding='utf-8') as f:
    lines = f.readlines()

# Переворачиваем
lines = lines[::-1]

# Находим хедер (он был в конце)
header_idx = 0
for i, line in enumerate(lines):
    if line.strip().startswith('timestamp'):
        header_idx = i
        break

header = lines[header_idx].strip().split(',')

# Читаем данные после заголовка
data_lines = lines[header_idx+1:]
data = [line.strip().split(',') for line in data_lines]

# Создаём DataFrame
df = pd.DataFrame(data, columns=header)

# Преобразуем нужные столбцы в числа
for col in ['min_actual_price', 'min_pred', 'hour_actual_price', 'hour_pred']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Считаем ошибки по минутным прогнозам
min_mask = df['min_actual_price'].notna() & df['min_pred'].notna()
min_errors = df.loc[min_mask, 'min_actual_price'] - df.loc[min_mask, 'min_pred']

min_mae = np.mean(np.abs(min_errors))
min_mse = np.mean(min_errors**2)

# Считаем ошибки по часовым прогнозам
hour_mask = df['hour_actual_price'].notna() & df['hour_pred'].notna()
hour_errors = df.loc[hour_mask, 'hour_actual_price'] - df.loc[hour_mask, 'hour_pred']

hour_mae = np.mean(np.abs(hour_errors))
hour_mse = np.mean(hour_errors**2)

print(f"Минутный прогноз: MAE = {min_mae:.2f}, MSE = {min_mse:.2f}")
print(f"Часовой прогноз: MAE = {hour_mae:.2f}, MSE = {hour_mse:.2f}")
