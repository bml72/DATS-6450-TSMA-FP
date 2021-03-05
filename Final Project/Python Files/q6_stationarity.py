import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
import numpy as np
import statsmodels.api as sm

# ======================================================================================================================
# Final Project: Checking for and/or making the dependent variable stationary
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

# ======================================================================================================================
# Plot of the dependent variable over time

plt.figure()

plt.plot(pm, label='PM 2.5')

plt.title('Plot of Dependent Variable (PM 2.5) vs Time')
plt.xlabel('Time (in hours)')
plt.ylabel('PM 2.5 Concentration (µg / cu m)')
plt.legend(loc='upper right')

plt.show()

# ======================================================================================================================
# Plot of the dependent variable over time (first month)

pm_first_month = pm.loc[:'2010-02-01']

plt.figure()

plt.plot(pm_first_month, label='PM 2.5')

plt.title('Plot of Dependent Variable (PM 2.5) vs Time (Jan \'10)')
plt.xlabel('Time (in hours)')
plt.ylabel('PM 2.5 Concentration (µg / cu m)')
plt.legend(loc='upper right')

plt.xticks(rotation=45)

plt.show()

print('Results of ADF test for raw data:')

adf_cal(pm)

# Analysis: Fails test; reject the null and conclude that data is stationary, but ACF and PACF say otherwise.

# ======================================================================================================================
# ACF plot of raw data

lags = 80

plt.figure()

plot_acf(pm, lags=lags)

plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('ACF of The Dependent Variable with {} Lags'.format(lags))

plt.show()

# ======================================================================================================================
# PACF plot of raw data

plt.figure()

plot_pacf(pm, lags=lags)

plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('PACF of The Dependent Variable with {} Lags'.format(lags))

plt.show()

# ======================================================================================================================
# First-order seasonal differencing

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
