from statsmodels.tsa.stattools import adfuller
import pandas as pd
from statsmodels.tsa.seasonal import STL
import numpy as np
import matplotlib.pyplot as plt

# ======================================================================================================================
# Final Project: Time series decomposition
# Benjamin Lee
# Data Science 6450-15: Time Series Modeling & Analysis
# 16 December 2020
# ======================================================================================================================
# Applicable functions


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
# Approximating the strength of the trend and seasonality

STL = STL(pm)
res = STL.fit()

trend = res.trend
seasonal = res.seasonal
residuals = res.resid

F_t = np.around(np.maximum(0, 1 - (np.var(np.array(residuals)) / np.var(np.array(trend + residuals)))), 5)
F_s = np.around(np.maximum(0, 1 - (np.var(np.array(residuals)) / np.var(np.array(seasonal + residuals)))), 5)

print('The strength of the trend is', F_t)
print('The strength of the seasonality is', F_s, '\n')

# ======================================================================================================================
# Initial assessment

# The results of the code above prove that the dependent variable is strongly trended and moderately seasonal.

# ======================================================================================================================
# Plotting the de-trended and seasonally adjusted data.

de_trended = pm - trend

seasonally_adjusted = pm - seasonal

plt.figure()

plt.plot(de_trended, label='De-Trended Dataset')
plt.plot(seasonally_adjusted, label='Seasonally-Adjusted Dataset')
plt.plot(pm, label='Original Data')

plt.xlabel('Time (in hours)')
plt.ylabel('PM 2.5 Concentration (µg/cu m)')
plt.title('Plot of De-trended and Seasonally-Adjusted Data')

plt.legend(loc='upper right', prop={'size': 8})

plt.xticks(rotation=45)

plt.show()

# ======================================================================================================================
# Assessment of plot

# The plot above does not provide significant insight into how the de-trended and seasonally-adjusted data compares to
# the raw data. Thus, the first month of data will be plotted to assess a smaller sample size.

# ======================================================================================================================
# Plotting the first month of de-trended and seasonally adjusted data.

pm_first_month = pm.loc[:'2010-02-01']

de_trended_first_month = de_trended.loc[:'2010-02-01']

seasonally_adjusted_first_month = seasonally_adjusted.loc[:'2010-02-01']

plt.figure()

plt.plot(de_trended_first_month, label='De-Trended Dataset')
plt.plot(seasonally_adjusted_first_month, label='Seasonally-Adjusted Dataset')
plt.plot(pm_first_month, label='Original Data')

plt.xlabel('Time (in hours)')
plt.ylabel('PM 2.5 Concentration (µg/cu m)')
plt.title('Plot of De-trended and Seasonally-Adjusted Data (Jan \'10)')

plt.legend(loc='upper right', prop={'size': 8})

plt.xticks(rotation=45)

plt.show()

# ======================================================================================================================
# Assessment of plot of first month

# The plot of de-trended and seasonally-adjusted data displays a near-perfect overlay of seasonally-adjusted data and
# the original dataset. The de-trended data lies below both the seasonally-adjusted and original dataset curves. These
# results suggest that the seasonal adjustments made to the dependent variable do not incite significant changes; this
# aligns with the results of the strengths of trend and seasonality. ADF tests for the de-trended data are run below.

# ======================================================================================================================
# Augmented Dickey-Fuller test

print('\nDe-trended data:\n')

adf_cal(de_trended)

print('\nDe-trended data (first month):\n')

adf_cal(de_trended_first_month)

print('\nSeasonally-adjusted data:\n')

adf_cal(seasonally_adjusted)

print('\nSeasonally-adjusted data (first month):\n')

adf_cal(seasonally_adjusted_first_month)

# ======================================================================================================================
# Assessment of ADF tests

# The ADF test for the raw de-trended data displays a large negative ADF statistic of -23.110642 and a p-value of
# 0.000000, while the ADF test for the first month of de-trended data displays a slightly smaller ADF statistic of
# -9.174005 and a p-value of 0.000000. The values of the ADF statistics and p-values suggest that the null hypothesis
# should be rejected and that in both cases, the dataset is stationary.
