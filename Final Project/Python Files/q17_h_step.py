import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import numpy as np

# ======================================================================================================================
# Final Project: Final model h-step prediction
# Benjamin Lee
# Data Science 6450-15: Time Series Modeling & Analysis
# 16 December 2020
# ======================================================================================================================
# Pre-processing procedures; see 'q5_description.py' for additional comments

df = pd.read_csv('PRSA_data_2010.1.1-2014.12.31.csv', header=0)

df['_date_'] = pd.to_datetime(df['year']*10000 + df['month']*100 + df['day'], format='%Y%m%d')

df['date'] = df['_date_'] + df['hour'].astype('timedelta64[h]')

df = df.drop(columns=['No', 'year', 'month', 'day', 'hour', '_date_'])

df = df.set_index('date')

df = df.loc['2010-01-02':'2011-01-01']

df = df.interpolate(method='time')

df['cbwd'] = df['cbwd'].astype('category')
df['cbwd'] = df['cbwd'].cat.codes

pm = df.loc[:, 'pm2.5']

predictors = df.drop(columns=['pm2.5', 'cbwd'])

predictors = sm.add_constant(predictors)

response = pm.copy(deep=True)

X_train, x_test, Y_train, y_test = train_test_split(predictors, response, shuffle=False, test_size=0.2)

# ======================================================================================================================
# Creating the SARIMA (2, 0, 0) x (1, 0, 0, 24) model

model = SARIMAX(endog=Y_train, order=(2, 0, 0), seasonal_order=(1, 0, 0, 24), freq='H').fit()

# ======================================================================================================================
# One-step ahead predictions

osa_predictions = []

for i in range(len(Y_train.index)):
    if i == 0:
        osa_predictions.append(model.params[0] * Y_train[i])
    elif 0 < i < 23:
        osa_predictions.append(model.params[0] * Y_train[i] + model.params[1] * Y_train[i - 1])
    elif i == 23:
        osa_predictions.append(model.params[0] * Y_train[i] + model.params[1] * Y_train[i - 1] + model.params[2] * Y_train[i - 23])
    elif i == 24:
        osa_predictions.append(model.params[0] * Y_train[i] + model.params[1] * Y_train[i - 1] + model.params[2] * Y_train[i - 23] - model.params[0] * model.params[2] * Y_train[i - 24])
    else:
        osa_predictions.append(model.params[0] * Y_train[i] + model.params[1] * Y_train[i - 1] + model.params[2] * Y_train[i - 23] - model.params[0] * model.params[2] * Y_train[i - 24] - model.params[1] * model.params[2] * Y_train[i - 25])

hsa_forecasts = []

for i in range(len(y_test)):
    if i == 0:
        hsa_forecasts.append(model.params[0] * Y_train[i] + model.params[1] * Y_train[i - 1] + model.params[2] * Y_train[i - 23] - model.params[0] * model.params[2] * Y_train[i - 24] - model.params[1] * model.params[2] * Y_train[i - 25])
    elif i == 1:
        hsa_forecasts.append(model.params[0] * osa_predictions[i - 1] + model.params[1] * Y_train[i - 1] + model.params[2] * Y_train[i - 22] - model.params[0] * model.params[2] * Y_train[i - 23] - model.params[1] * model.params[2] * Y_train[i - 24])
    elif 1 < i < 25:
        hsa_forecasts.append(model.params[0] * osa_predictions[i - 1] + model.params[1] * osa_predictions[i - 2] + model.params[2] * Y_train[i - (23 - i)] - model.params[0] * model.params[2] * Y_train[i - (24 - i)] - model.params[1] * model.params[2] * Y_train[i - (25 - i)])
    elif i == 25:
        hsa_forecasts.append(model.params[0] * osa_predictions[i - 1] + model.params[1] * osa_predictions[i - 2] + model.params[2] * osa_predictions[i - 25] - model.params[0] * model.params[2] * Y_train[i - (24 - i)] - model.params[1] * model.params[2] * Y_train[i - (25 - i)])
    elif i == 26:
        hsa_forecasts.append(model.params[0] * osa_predictions[i - 1] + model.params[1] * osa_predictions[i - 2] + model.params[2] * osa_predictions[i - 25] - model.params[0] * model.params[2] * osa_predictions[i - 26] - model.params[1] * model.params[2] * Y_train[i - 26])
    else:
        hsa_forecasts.append(model.params[0] * osa_predictions[i - 1] + model.params[1] * osa_predictions[i - 2] + model.params[2] * osa_predictions[i - 25] - model.params[0] * model.params[2] * osa_predictions[i - 26] - model.params[1] * model.params[2] * osa_predictions[i - 27])

# ======================================================================================================================
# Plotting the h-step ahead forecasts

osa_predictions = pd.Series(osa_predictions, index=Y_train.index)

hsa_forecasts = pd.Series(hsa_forecasts, index=y_test.index)

plt.figure()

plt.plot(Y_train, label='Training data')
plt.plot(y_test, label='Testing data')
plt.plot(osa_predictions, label='One-step ahead predictions')
plt.plot(hsa_forecasts, label='H-step ahead forecasts')

plt.title('Plot of OSA predictions - SARIMA (2, 0, 0) x (1, 0, 0, 24)')
plt.xlabel('Time (in hours)')
plt.ylabel('PM 2.5 Concentration (µg / cu m)')
plt.legend(loc='upper left')

plt.show()

# ======================================================================================================================
# Variance of the forecast errors

forecast_errors = y_test - hsa_forecasts

print('The variance of the forecast errors is', np.var(forecast_errors))
