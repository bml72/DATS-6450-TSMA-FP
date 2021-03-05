import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# ======================================================================================================================
# Final Project: Base models
# Benjamin Lee
# Data Science 6450-15: Time Series Modeling & Analysis
# 16 December 2020
# ======================================================================================================================
# Applicable functions


def one_step_ahead_average_method(training_data):
    """
    :param training_data: The training set used in this assignment.
    :return: A list containing the one-step ahead predictions using the average forecast method.
    """
    predictions = [0]
    for value in range(len(training_data) - 1):
        predictions.append(round(sum(training_data[:value + 1]) / (value + 1), 3))
    return predictions


def h_step_ahead_average_method(training_data, testing_data):
    """
    :param training_data: The training set used in this assignment.
    :param testing_data: The testing set used in this assignment.
    :return: A list containing the h-step ahead forecasts using the average forecast method.
    """
    forecast_value = round(sum(training_data[::]) / len(training_data), 3)
    forecasts = [forecast_value] * len(testing_data)
    return forecasts


def one_step_ahead_naive_method(training_data):
    """
    :param training_data: The training set used in this assignment.
    :return: A list containing the one-step ahead predictions using the naive forecast method.
    """
    predictions = [0]
    for value in range(len(training_data) - 1):
        predictions.append(training_data[value])
    return predictions


def h_step_ahead_naive_method(training_data, testing_data):
    """
    :param training_data: The training set used in this assignment.
    :param testing_data: The testing set used in this assignment.
    :return: A list containing the h-step ahead forecasts using the average forecast method.
    """
    forecast_value = training_data[-1]
    forecasts = [forecast_value] * len(testing_data)
    return forecasts


def one_step_ahead_drift_method(training_data):
    """
    :param training_data: The training set used in this assignment.
    :return: A list containing the one-step ahead forecasts using the drift forecast method.
    """
    predictions = [0, training_data[0]]
    for value in range(1, len(training_data) - 1):
        predictions.append(round(training_data[value] + ((training_data[value] - training_data[0]) / value), 3))
    return predictions


def h_step_ahead_drift_method(training_data, testing_data):
    """
    :param training_data: The training set used in this assignment.
    :param testing_data: The testing set used in this assignment.
    :return: A list containing the h-step ahead forecasts using the drift forecast method.
    """
    forecasts = []
    for value in range(1, len(testing_data) + 1):
        forecasts.append(round(training_data[-1] + value * ((training_data[-1] - training_data[0]) / (len(training_data) - 1)), 3))
    return forecasts


def one_step_ahead_ses_method(training_data, alpha):
    """
    :param training_data: The training set used in this assignment.
    :param alpha: The value of the smoothing parameter.
    :return: A list containing the one-step ahead forecasts using the simple exponential smoothing forecast method.
    """
    predictions = [training_data[0]]
    for value in range(len(training_data) - 1):
        predictions.append(round(alpha * training_data[value] + (1 - alpha) * predictions[value], 3))
    return predictions


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

lags = 50

# ======================================================================================================================
# Average method

osa_predictions_average_method = one_step_ahead_average_method(Y_train)
osa_predictions_average_method = pd.Series(osa_predictions_average_method, index=Y_train.index)

hsa_forecasts_average_method = h_step_ahead_average_method(Y_train, y_test)
hsa_forecasts_average_method = pd.Series(hsa_forecasts_average_method, index=y_test.index)

plt.figure()

plt.plot(Y_train, label='Training Dataset')
plt.plot(y_test, label='Testing Dataset')
plt.plot(osa_predictions_average_method, label='One-step ahead predictions\n(Average Method)')
plt.plot(hsa_forecasts_average_method, label='H-step ahead predictions\n(Average Method)')

plt.title('Plot of Average method predictions and forecasts vs original data')
plt.xlabel('Time (in hours)')
plt.ylabel('PM 2.5 Concentration (µg / cu m)')
plt.legend(loc='upper right')

plt.show()

plt.figure()

plt.plot(Y_train[:50], label='Training Dataset')
plt.plot(osa_predictions_average_method[:50], label='One-step ahead predictions\n(Average Method)')

plt.title('Plot of Average method vs original data (first 50 samples)')
plt.xlabel('Time (in hours)')
plt.ylabel('PM 2.5 Concentration (µg / cu m)')
plt.legend(loc='upper right')

plt.xticks(rotation=45)

plt.show()

# Residuals

residuals = []

for i in range(len(Y_train.index)):
    residuals.append(Y_train[i] - osa_predictions_average_method[i])

print('The mean of the residuals for the average method are', np.mean(residuals))

# Forecast errors

forecast_errors = []

for i in range(len(y_test.index)):
    forecast_errors.append(y_test[i] - hsa_forecasts_average_method[i])

print('\nThe variance of the residuals for the average method is', np.var(residuals))
print('The variance of the forecast errors for the average method is', np.var(forecast_errors))

re = sm.tsa.stattools.acf(residuals, nlags=lags)

Q = len(Y_train) * np.sum(np.square(re))

print('The Q-value of the residuals is', Q)

# Analysis: The mean of the residuals is somewhat close to zero (much closer than the SARIMA models), but is still
# likely a biased estimator due to its non-zero value. Additionally, the overall curves of the one-step ahead
# predictions and H-step ahead forecasts for the average are terrible in comparison to the original training set. The
# variance of the forecast errors is much higher than the variance of the one-step ahead predictions. This is likely
# not the best model for this dataset. Additionally, Q-value is enormous; the auto-correlations definitely don't come
# from a white noise series.

# ======================================================================================================================
# Naive method

osa_predictions_naive_method = one_step_ahead_naive_method(Y_train)
osa_predictions_naive_method = pd.Series(osa_predictions_naive_method, index=Y_train.index)

hsa_forecasts_naive_method = h_step_ahead_naive_method(Y_train, y_test)
hsa_forecasts_naive_method = pd.Series(hsa_forecasts_naive_method, index=y_test.index)

plt.figure()

plt.plot(Y_train, label='Training Dataset')
plt.plot(y_test, label='Testing Dataset')
plt.plot(osa_predictions_naive_method, label='One-step ahead predictions\n(Naive Method)')
plt.plot(hsa_forecasts_naive_method, label='H-step ahead predictions\n(Naive Method)')

plt.title('Plot of Naive method predictions and forecasts vs original data')
plt.xlabel('Time (in hours)')
plt.ylabel('PM 2.5 Concentration (µg / cu m)')
plt.legend(loc='upper right')

plt.show()

plt.figure()

plt.plot(Y_train[:50], label='Training Dataset')
plt.plot(osa_predictions_naive_method[:50], label='One-step ahead predictions\n(Naive Method)')

plt.title('Plot of Naive method vs original data (first 50 samples)')
plt.xlabel('Time (in hours)')
plt.ylabel('PM 2.5 Concentration (µg / cu m)')
plt.legend(loc='upper right')

plt.xticks(rotation=45)

plt.show()

# Residuals

residuals = []

for i in range(len(Y_train.index)):
    residuals.append(Y_train[i] - osa_predictions_naive_method[i])

print('\nThe mean of the residuals for the naive method are', np.mean(residuals))

# Forecast errors

forecast_errors = []

for i in range(len(y_test.index)):
    forecast_errors.append(y_test[i] - hsa_forecasts_naive_method[i])

print('\nThe variance of the residuals for the naive method is', np.var(residuals))
print('The variance of the forecast errors for the naive method is', np.var(forecast_errors))

re = sm.tsa.stattools.acf(residuals, nlags=lags)

Q = len(Y_train) * np.sum(np.square(re))

print('The Q-value of the residuals is', Q)

# Analysis: The naive method performs great on the training set; the values of the training set and the one-step ahead
# predictions practically overlap. Additionally, the mean of the residuals is very close to zero; this means the naive
# model is likely an unbiased estimator. However, the results of the h-step ahead predictions are awful. The h-step
# ahead predictions do not match the testing set at all, and the variance of the forecast errors are extremely large.
# This is likely not the best model for this dataset. The Q-value is large, but right on par with the SARIMA methods;
# the auto-correlations likely don't come from a white noise series.

# ======================================================================================================================
# Drift method

osa_predictions_drift_method = one_step_ahead_drift_method(Y_train)
osa_predictions_drift_method = pd.Series(osa_predictions_drift_method, index=Y_train.index)

hsa_forecasts_drift_method = h_step_ahead_drift_method(Y_train, y_test)
hsa_forecasts_drift_method = pd.Series(hsa_forecasts_drift_method, index=y_test.index)

plt.figure()

plt.plot(Y_train, label='Training Dataset')
plt.plot(y_test, label='Testing Dataset')
plt.plot(osa_predictions_drift_method, label='One-step ahead predictions\n(Drift Method)')
plt.plot(hsa_forecasts_drift_method, label='H-step ahead predictions\n(Drift Method)')

plt.title('Plot of Drift method predictions and forecasts vs original data')
plt.xlabel('Time (in hours)')
plt.ylabel('PM 2.5 Concentration (µg / cu m)')
plt.legend(loc='upper right')

plt.show()

plt.figure()

plt.plot(Y_train[:50], label='Training Dataset')
plt.plot(osa_predictions_drift_method[:50], label='One-step ahead predictions\n(Drift Method)')

plt.title('Plot of Drift method vs original data (first 50 samples)')
plt.xlabel('Time (in hours)')
plt.ylabel('PM 2.5 Concentration (µg / cu m)')
plt.legend(loc='upper right')

plt.xticks(rotation=45)

plt.show()

# Residuals

residuals = []

for i in range(len(Y_train.index)):
    residuals.append(Y_train[i] - osa_predictions_drift_method[i])

print('\nThe mean of the residuals for the Drift method are', np.mean(residuals))

# Forecast errors

forecast_errors = []

for i in range(len(y_test.index)):
    forecast_errors.append(y_test[i] - hsa_forecasts_drift_method[i])

print('\nThe variance of the residuals for the Drift method is', np.var(residuals))
print('The variance of the forecast errors for the Drift method is', np.var(forecast_errors))

re = sm.tsa.stattools.acf(residuals, nlags=lags)

Q = len(Y_train) * np.sum(np.square(re))

print('The Q-value of the residuals is', Q)

# Analysis: The results of the drift method are very similar to the naive method. The only big change is the h-step
# ahead forecasts, whose plot displays a slightly downward trend. The h-step ahead forecasts using the drift method are
# terrible; this is also likely not the best model for the dataset. The Q-value is large, but right on par with the
# SARIMA methods; the auto-correlations likely don't come from a white noise series.

# ======================================================================================================================
# Simple exponential smoothing

osa_predictions_ses_method = one_step_ahead_ses_method(Y_train, alpha=0.5)
osa_predictions_ses_method = pd.Series(osa_predictions_ses_method, index=Y_train.index)

hsa_forecasts_ses_method = [osa_predictions_ses_method[-1] for _ in range(len(y_test))]
hsa_forecasts_ses_method = pd.Series(hsa_forecasts_ses_method, index=y_test.index)

plt.figure()

plt.plot(Y_train, label='Training Dataset')
plt.plot(y_test, label='Testing Dataset')
plt.plot(osa_predictions_ses_method, label='One-step ahead predictions\n(SES Method)')
plt.plot(hsa_forecasts_ses_method, label='H-step ahead predictions\n(SES Method)')

plt.title('Plot of Simple Exponential Smoothing vs original data')
plt.xlabel('Time (in hours)')
plt.ylabel('PM 2.5 Concentration (µg / cu m)')
plt.legend(loc='upper right')

plt.show()

plt.figure()

plt.plot(Y_train[:50], label='Training Dataset')
plt.plot(osa_predictions_ses_method[:50], label='One-step ahead predictions\n(SES Method)')

plt.title('Plot of SES Method vs original data (first 50 samples)')
plt.xlabel('Time (in hours)')
plt.ylabel('PM 2.5 Concentration (µg / cu m)')
plt.legend(loc='upper right')

plt.xticks(rotation=45)

plt.show()

# Residuals

residuals = []

for i in range(len(Y_train.index)):
    residuals.append(Y_train[i] - osa_predictions_ses_method[i])

print('\nThe mean of the residuals for the SES method are', np.mean(residuals))

# Forecast errors

forecast_errors = []

for i in range(len(y_test.index)):
    forecast_errors.append(y_test[i] - hsa_forecasts_ses_method[i])

print('\nThe variance of the residuals for the SES method is', np.var(residuals))
print('The variance of the forecast errors for the SES method is', np.var(forecast_errors))

re = sm.tsa.stattools.acf(residuals, nlags=lags)

Q = len(Y_train) * np.sum(np.square(re))

print('The Q-value of the residuals is', Q)

# Analysis: Simple exponential works very well on the training set, and the mean of the residuals is practically zero.
# However, the plot of h-step ahead predictions vs the testing set is terrible. h-step ahead predictions do not match
# the testing set at all. This is likely not a good model for this dataset. The Q-value is a bit larger than the other
# base models, as well as the SARIMA methods; the auto-correlations likely don't come from a white noise series.
