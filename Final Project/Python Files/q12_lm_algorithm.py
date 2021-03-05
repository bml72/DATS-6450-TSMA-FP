import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt


# ======================================================================================================================
# Final Project: Levenberg Marquardt algorithm
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

# ======================================================================================================================
# Confidence intervals and parameter estimates

print(100*'-')

sarima_model_1_conf_int = model_1.conf_int()
sarima_model_1_conf_int = sarima_model_1_conf_int.values.tolist()
sarima_model_1_conf_int = [val for sublist in sarima_model_1_conf_int for val in sublist]

print('The confidence intervals of SARIMA (2, 0, 0) x (1, 0, 0, 24) are as follows:\n')
print('Coefficient a1:', sarima_model_1_conf_int[0], '<=', model_1.params[0], '<=', sarima_model_1_conf_int[1])
print('Coefficient a2:', sarima_model_1_conf_int[2], '<=', model_1.params[1], '<=', sarima_model_1_conf_int[3])
print('Coefficient a24:', sarima_model_1_conf_int[4], '<=', model_1.params[2], '<=', sarima_model_1_conf_int[5])

sarima_model_2_conf_int = model_2.conf_int()
sarima_model_2_conf_int = sarima_model_2_conf_int.values.tolist()
sarima_model_2_conf_int = [val for sublist in sarima_model_2_conf_int for val in sublist]

print('The confidence intervals of SARIMA (2, 0, 0) x (2, 0, 0, 24) are as follows:\n')
print('Coefficient a1:', sarima_model_2_conf_int[0], '<=', model_2.params[0], '<=', sarima_model_2_conf_int[1])
print('Coefficient a2:', sarima_model_2_conf_int[2], '<=', model_2.params[1], '<=', sarima_model_2_conf_int[3])
print('Coefficient a24:', sarima_model_2_conf_int[4], '<=', model_2.params[2], '<=', sarima_model_2_conf_int[5])
print('Coefficient a48:', sarima_model_2_conf_int[6], '<=', model_2.params[3], '<=', sarima_model_2_conf_int[7])

sarima_model_3_conf_int = model_3.conf_int()
sarima_model_3_conf_int = sarima_model_3_conf_int.values.tolist()
sarima_model_3_conf_int = [val for sublist in sarima_model_3_conf_int for val in sublist]

print('The confidence intervals of SARIMA (1, 1, 0) x (0, 1, 1, 24) are as follows:\n')
print('Coefficient a1:', sarima_model_3_conf_int[0], '<=', model_3.params[0], '<=', sarima_model_3_conf_int[1])
print('Coefficient ma24:', sarima_model_3_conf_int[2], '<=', model_3.params[1], '<=', sarima_model_3_conf_int[3])

# ======================================================================================================================
# Standard deviation of the estimates

# This section is calculated by taking the difference of the coefficient and the upper confidence interval (to find the
# value of the confidence interval) and divide the result by two; this follows the equation for confidence interval in
# the lecture 10 slides. The entire square root of the equation is the standard deviation of the coefficient.

print('\nStandard deviations:')

# SARIMA (2, 0, 0) x (1, 0, 0, 24)

print('\nSARIMA (2, 0, 0) x (1, 0, 0, 24):\n')

print('Coefficient a1:', (1.134 - 1.1280) / 2)
print('Coefficient a2:', (-0.146 + 0.1525) / 2)
print('Coefficient a24:', (0.031 - 0.0070) / 2)

# SARIMA (2, 0, 0) x (2, 0, 0, 24)

print('\nSARIMA (2, 0, 0) x (2, 0, 0, 24):\n')

print('Coefficient a1:', (1.131 - 1.1251) / 2)
print('Coefficient a2:', (-0.146 + 0.1516) / 2)
print('Coefficient a24:', (0.033 - 0.0084) / 2)
print('Coefficient a48:', (0.077 - 0.0645) / 2)

# SARIMA (1, 1, 0) x (0, 1, 1, 24)

print('\nSARIMA (1, 1, 0) x (0, 1, 1, 24):\n')

print('Coefficient a1:', (0.140 - 0.1347) / 2)
print('Coefficient ma24:', (-0.971 + 0.9985))
