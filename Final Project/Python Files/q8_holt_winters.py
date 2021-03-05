import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.tsa.holtwinters as ets
import matplotlib.pyplot as plt

# ======================================================================================================================
# Final Project: Holt-Winters method
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

predictors = df.drop(columns='pm2.5')
response = pm.copy(deep=True)

X_train, x_test, Y_train, y_test = train_test_split(predictors, response, test_size=0.2, shuffle=False)

# ======================================================================================================================
# Holt-Winters Method

holt_winters_training = ets.ExponentialSmoothing(Y_train, trend='mul', damped_trend=True, seasonal='add',
                                                 seasonal_periods=7, freq='H').fit()

holt_winters_forecast = holt_winters_training.forecast(steps=len(y_test))
holt_winters_forecast = pd.DataFrame(holt_winters_forecast).set_index(y_test.index)

fig, ax = plt.subplots()

ax.plot(Y_train, label='Training Data')
ax.plot(y_test, label='Testing Data')
ax.plot(holt_winters_forecast, label='Holt-Winters Method')

plt.title('Plot of Holt-Winters Method vs Training & Testing Data')
plt.xlabel('Time (in hours)')
plt.ylabel('PM 2.5 Concentration (µg/cu m)')
plt.legend(loc='upper right')

plt.show()

# ======================================================================================================================
# Analysis of plot

# After testing Holt-Winters method with a variety of combinations for the trend and seasonal values, the best forecast
# resulted in setting trend to multiplicative and seasonal to additive with a seasonal period of 7 (see
# 'q5_description.py' for reasoning behind seasonal period). I also tried plotting Holt's Linear Trend, but the
# forecasts for both options of trend were very weak.

# ======================================================================================================================
# Holt's Linear Trend using multiplicative trend

holt_linear_training = ets.ExponentialSmoothing(Y_train, trend='mul', damped_trend=True, seasonal=None, freq='H').fit()

holt_linear_forecast = holt_linear_training.forecast(steps=len(y_test))
holt_linear_forecast = pd.DataFrame(holt_linear_forecast).set_index(y_test.index)

fig, ax = plt.subplots()

ax.plot(Y_train, label='Training Data')
ax.plot(y_test, label='Testing Data')
ax.plot(holt_linear_forecast, label='Holt\'s Linear Trend Method')

plt.title('Plot of Holt\'s Linear Trend vs Training & Testing Data')
plt.xlabel('Time (in hours)')
plt.ylabel('PM 2.5 Concentration (µg/cu m)')
plt.legend(loc='upper right')

plt.show()

# ======================================================================================================================
