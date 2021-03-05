import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import seaborn as sns
from sklearn.model_selection import train_test_split

# ======================================================================================================================
# Final Project: Description of the dataset
# Benjamin Lee
# Data Science 6450-15: Time Series Modeling & Analysis
# 16 December 2020
# ======================================================================================================================
# Part D: Pre-processing procedures

# Reading in the dataset.
df = pd.read_csv('PRSA_data_2010.1.1-2014.12.31.csv', header=0)

# Combining the time-related columns to a single datetime column.
df['_date_'] = pd.to_datetime(df['year']*10000 + df['month']*100 + df['day'], format='%Y%m%d')

# Converting the hours to time-deltas and combining with the datetime column.
df['date'] = df['_date_'] + df['hour'].astype('timedelta64[h]')

# Dropping all irrelevant columns; this includes the index column 'No' and the old time-related columns.
df = df.drop(columns=['No', 'year', 'month', 'day', 'hour', '_date_'])

# Setting the index as the newly-created datetime column.
df = df.set_index('date')

# Excluding the first day (24 samples) of data. As per section 12.9 of the textbook:
#
# When missing values cause errors, there are at least two ways to handle the problem. First, we could just take the
# section of data after the last missing value, assuming there is a long enough series of observations to produce
# meaningful forecasts. Alternatively, we could replace the missing values with estimates. The na.interp() function
# is designed for this purpose.
#
# Source: https://otexts.com/fpp2/missing-outliers.html

# The equivalent function in Python, pd.interpolate(), cannot interpolate the first x samples of a dataset because there
# are no entries before them to use for interpolation.
#
# Source: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html
df = df.loc['2010-01-02':'2011-01-01']

# Interpolating using the 'time' method, since this data is based over time.
df = df.interpolate(method='time')

# Label encoding the combined wind direction variable; NE = 0, NW = 1, SW = 2 and cv = 3.
df['cbwd'] = df['cbwd'].astype('category')
df['cbwd'] = df['cbwd'].cat.codes
# 'cv' is a reference to 'calm and variable.'
#
# Source: https://royalsocietypublishing.org/doi/10.1098/rspa.2015.0257

# Initializing the dependent variable.
pm = df.loc[:, 'pm2.5']

# ======================================================================================================================
# Part A: Plotting the dependent variable versus time

plt.figure()

plt.plot(pm, label='PM 2.5')

plt.title('Plot of Dependent Variable (PM 2.5) vs Time')
plt.xlabel('Time (in hours)')
plt.ylabel('PM 2.5 Concentration (µg / cu m)')
plt.legend(loc='upper right')

plt.show()

# ======================================================================================================================
# Part B: Plotting the ACF and PACF of the dependent variable

lags = 100

plt.figure()

plot_acf(pm, lags=lags)

plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('ACF of The Dependent Variable with {} Lags'.format(lags))

plt.show()

plt.figure()

plot_pacf(pm, lags=lags)

plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('PACF of The Dependent Variable with {} Lags'.format(lags))

plt.show()

# ======================================================================================================================
# Part C: Plotting the correlation matrix of the dataset using Pearson's correlation coefficient

plt.figure()

ax = sns.heatmap(df.corr(method='pearson'), annot=True, fmt='.3g', vmin=-1, vmax=1, center=0, cmap='Blues',
                 linecolor='black', square=True, annot_kws={'size': 7})

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='right', fontsize=9)

plt.title('HeatMap of Correlation Matrix')

plt.show()

# ======================================================================================================================
# Part E: Splitting the dataset into training and testing

predictors = df.drop(columns='pm2.5')
response = pm.copy(deep=True)

X_train, x_test, Y_train, y_test = train_test_split(predictors, response, test_size=0.2, shuffle=False)

# ======================================================================================================================
# Additional analysis: plotting first month of dependent variable vs time

# NOTE: It is quite difficult to visualize seasonality in the dataset using all of the raw data. The first month of data
# is plotted below to get a better sense of what's going on.

pm_first_month = pm.loc[:'2010-02-01']

plt.figure()

plt.plot(pm_first_month, label='PM 2.5')

plt.title('Plot of Dependent Variable (PM 2.5) vs Time (Jan \'10)')
plt.xlabel('Time (in hours)')
plt.ylabel('PM 2.5 Concentration (µg / cu m)')
plt.legend(loc='upper right')

plt.xticks(rotation=45)

plt.show()

# The seasonal period appears to be roughly 7; the large spikes occur on Jan 2, Jan 8/9, Jan 14, Jan 20, Jan 27 and
# Jan 31.
