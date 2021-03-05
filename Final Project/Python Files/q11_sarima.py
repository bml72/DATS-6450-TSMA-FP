import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ======================================================================================================================
# Final Project: Order estimation of SARIMA
# Benjamin Lee
# Data Science 6450-15: Time Series Modeling & Analysis
# 16 December 2020
# ======================================================================================================================
# Applicable functions


def difference(dataset, interval):
    diff = []
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return diff


def adf_cal(x):
    """
    :param x: The dataset in question.
    :return: The results of the Augmented Dickey-Fuller Test.
    """
    result = adfuller(x)

    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, val in result[4].items():
        print('\t%s: %.3f' % (key, val))


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
# First-order seasonal differencing

lags = 80

pm_seasonal = difference(dataset=pm, interval=24)

# pm_first_difference = difference(dataset=pm_seasonal, interval=1)
#
# acf = sm.tsa.stattools.acf(pm_first_difference, nlags = lags)
# pacf = sm.tsa.stattools.pacf(pm_first_difference, nlags = lags)

# ======================================================================================================================
# ACF plot of first-order seasonal differencing

plt.figure()

plot_acf(pm_seasonal, lags=lags)

plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('ACF of First-Order Seasonal Differencing with {} Lags'.format(lags))

plt.show()

# ======================================================================================================================
# PACF plot of first-order seasonal differencing

plt.figure()

plot_pacf(pm_seasonal, lags=lags)

plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('PACF of First-Order Seasonal Differencing with {} Lags'.format(lags))

plt.show()

# ======================================================================================================================
# First-order non-seasonal differencing

pm_first_difference = difference(dataset=pm_seasonal, interval=1)

# ======================================================================================================================
# ACF plot of first-order non-seasonal differencing

plt.figure()

plot_acf(pm_first_difference, lags=lags)

plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('ACF of First-Order Non-Seasonal Differencing with {} Lags'.format(lags))

plt.show()

# ======================================================================================================================
# PACF of first-order non-seasonal differencing

plt.figure()

plot_pacf(pm_first_difference, lags=lags)

plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('PACF of First-Order Non-Seasonal Differencing with {} Lags'.format(lags))

plt.show()

# ======================================================================================================================
# ADF testing

print('ADF test of first-order seasonal differencing, followed by first-order non-seasonal differencing:')

adf_cal(pm_first_difference)

# Analysis: still reject the null and conclude data is stationary.

# ======================================================================================================================
# The plots of the ACF and PACF denote the following:

# First-order seasonal differencing: ACF cuts off at lag 24, PACF tailing off every 24 lags
# First-order non-seasonal differencing: ACF and PACF cut off every 24 lags.

# Conclusion: Order estimates are a combination of AR/MA seasonal and non-seasonal; after office hours,
# concluded to try the following:

# SARIMA (2, 0, 0) x (1, 0, 0, 24)
# SARIMA (2, 0, 0) x (2, 0, 0, 24)
# SARIMA (1, 1, 0) x (0, 1, 1, 24)

# ======================================================================================================================
# Creating the SARIMA models

model_1 = SARIMAX(endog=Y_train, order=(2, 0, 0), seasonal_order=(1, 0, 0, 24), freq='H').fit()

print('Summary of SARIMA (2, 0, 0) x (1, 0, 0, 24):\n')

print(model_1.summary())

model_2 = SARIMAX(endog=Y_train, order=(2, 0, 0), seasonal_order=(2, 0, 0, 24), freq='H').fit()

print('Summary of SARIMA (2, 0, 0) x (2, 0, 0, 24):\n')

print(model_2.summary())

model_3 = SARIMAX(endog=Y_train, order=(1, 1, 0), seasonal_order=(0, 1, 1, 24), freq='H').fit()

print('Summary of SARIMA (1, 1, 0) x (0, 1, 1, 24):\n')

print(model_3.summary())
