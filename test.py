import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.0, 8.0])

errors = y_true - y_pred
mse_manual = np.mean(errors ** 2)     # 0.375
mae_manual = np.mean(np.abs(errors)) # 0.5

mse_sklearn = mean_squared_error(y_true, y_pred)     # 0.375
mae_sklearn = mean_absolute_error(y_true, y_pred)    # 0.5
