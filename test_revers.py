import pandas as pd
import matplotlib.pyplot as plt
import pytz
import numpy as np

# Путь к файлу
predictions_file = "logs/smol.csv"

# ---------- ЧТЕНИЕ И ПРАВИЛЬНАЯ ЗАГРУЗКА ----------

# Считываем весь файл построчно
with open(predictions_file, encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

# Находим строку с названиями столбцов
header_line = None
for i, line in enumerate(lines):
    if line.lower().startswith("timestamp"):
        header_line = line
        header_index = i
        break

if header_line is None:
    raise ValueError("В файле не найден заголовок со словом 'timestamp'!")

# Формируем DataFrame из всех строк выше заголовка
data_lines = lines[:header_index]

# Добавляем найденный заголовок наверх
corrected_lines = [header_line] + data_lines

# Читаем как CSV
from io import StringIO
csv_content = "\n".join(corrected_lines)
df = pd.read_csv(StringIO(csv_content))

# ---------- ОБРАБОТКА ВРЕМЕНИ ----------

# Преобразуем строки в datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour_pred_time"] = pd.to_datetime(df["hour_pred_time"])

# Сортировка по времени (на случай перевёрнутого файла)
df = df.sort_values("timestamp").reset_index(drop=True)

# Установка часового пояса (MSK)
msk_tz = pytz.timezone("Europe/Moscow")
df["timestamp"] = df["timestamp"].dt.tz_convert(msk_tz)
df["hour_pred_time"] = df["hour_pred_time"].dt.tz_convert(msk_tz)

# ---------- ФИЛЬТРАЦИЯ ОФФЛАЙН-ДАННЫХ ----------

df_offline = df[df["hour_pred_time"].dt.minute == 0]
df_offline = df_offline.sort_values("hour_pred_time").reset_index(drop=True)

# ---------- РАСЧЁТ MAE ----------

online_mae = df["hour_error"].mean()
offline_mae = df_offline["hour_error"].mean() if not df_offline.empty else np.nan

# ---------- ВИЗУАЛИЗАЦИЯ ----------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

# Онлайн-график
ax1.plot(df["timestamp"], df["actual_price"], label="Actual Price", color="blue")
ax1.plot(df["timestamp"], df["hour_pred"], label=f"Hourly Prediction (MAE={online_mae:.2f})", color="orange", linestyle="--")
ax1.set_title("Online Version (Hourly Predictions Every Minute)")
ax1.set_xlabel("Timestamp")
ax1.set_ylabel("Price")
ax1.legend()
ax1.grid(True)
ax1.tick_params(axis="x", rotation=45)

# Оффлайн-график
ax2.plot(df_offline["hour_pred_time"], df_offline["hour_actual_price"], label="Actual Price", color="blue")
ax2.plot(df_offline["hour_pred_time"], df_offline["hour_pred"], label=f"Hourly Prediction (MAE={offline_mae:.2f})", color="orange", linestyle="--")
ax2.set_title("Offline Version (Hourly Predictions at Hour Start)")
ax2.set_xlabel("Hour Prediction Time")
ax2.set_ylabel("Price")
ax2.legend()
ax2.grid(True)
ax2.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()

# ---------- СТАТИСТИКА ----------

print(f"Online MAE (hourly): {online_mae:.2f}")
print(f"Offline MAE (hourly): {offline_mae:.2f}")
print(f"Number of predictions (online): {len(df)}")
print(f"Number of predictions (offline): {len(df_offline)}")
