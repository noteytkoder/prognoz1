import pandas as pd
import matplotlib.pyplot as plt
import pytz
import numpy as np

# Путь к файлу с предсказаниями
predictions_file = "logs/predictions.csv"

# Чтение данных
df = pd.read_csv(predictions_file)

# Преобразование временных столбцов в datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour_pred_time"] = pd.to_datetime(df["hour_pred_time"])

# Установка временной зоны (MSK)
msk_tz = pytz.timezone("Europe/Moscow")
df["timestamp"] = df["timestamp"].dt.tz_convert(msk_tz)
df["hour_pred_time"] = df["hour_pred_time"].dt.tz_convert(msk_tz)

# Фильтрация для оффлайн-версии: берем только предсказания на начало часа
df_offline = df[df["hour_pred_time"].dt.minute == 0]

# Расчёт MAE для онлайн- и оффлайн-версий
online_mae = df["hour_error"].mean()
offline_mae = df_offline["hour_error"].mean() if not df_offline.empty else np.nan

# Создание фигуры с двумя подграфиками
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

# График для онлайн-версии (все предсказания каждую минуту)
ax1.plot(df["timestamp"], df["actual_price"], label="Actual Price", color="blue")
ax1.plot(df["timestamp"], df["hour_pred"], label=f"Hourly Prediction (MAE={online_mae:.2f})", color="orange", linestyle="--")
ax1.set_title("Online Version (Hourly Predictions Every Minute)")
ax1.set_xlabel("Timestamp")
ax1.set_ylabel("Price")
ax1.legend()
ax1.grid(True)
ax1.tick_params(axis="x", rotation=45)

# График для оффлайн-версии (предсказания только на начало часа)
ax2.plot(df_offline["hour_pred_time"], df_offline["hour_actual_price"], label="Actual Price", color="blue")
ax2.plot(df_offline["hour_pred_time"], df_offline["hour_pred"], label=f"Hourly Prediction (MAE={offline_mae:.2f})", color="orange", linestyle="--")
ax2.set_title("Offline Version (Hourly Predictions at Hour Start)")
ax2.set_xlabel("Hour Prediction Time")
ax2.set_ylabel("Price")
ax2.legend()
ax2.grid(True)
ax2.tick_params(axis="x", rotation=45)

# Настройка отображения
plt.tight_layout()
plt.show()

# Вывод статистики
print(f"Online MAE (hourly): {online_mae:.2f}")
print(f"Offline MAE (hourly): {offline_mae:.2f}")
print(f"Number of predictions (online): {len(df)}")
print(f"Number of predictions (offline): {len(df_offline)}")