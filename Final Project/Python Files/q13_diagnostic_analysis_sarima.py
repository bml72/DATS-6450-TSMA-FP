import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt

# ======================================================================================================================
# Final Project: Diagnostic analysis of SARIMA processes
# Benjamin Lee
# Data Science 6450-15: Time Series Modeling & Analysis
# 16 December 2020
# ======================================================================================================================
# Applicable functions


def difference(dataset, interval):
    """
    :param dataset: The dataset in question.
    :param interval: The interval for differencing. If seasonal differencing, enter the seasonality period here. For
    # simple differencing (non-seasonal), enter the order of differencing preferred.
    :return: The differenced data. Credit is given to professor Jafari, who provided this function in email
    # correspondence.
    """
    diff = []
    for i in range(interval, len(dataset)):
        _value_ = dataset[i] - dataset[i - interval]
        diff.append(_value_)
    return diff


def acf_function(dataset, lags):
    """
    :param dataset: The data used in calculating the ACF values.
    :param lags: The number of lags specified for calculating the ACF values.
    :return: The ACF values associated with the given dataset at a specified lag value.
    # Values for t are adjusted to account for the indexing differences between Python and the handwritten ACF.
    """
    t = lags + 1

    max_t = len(dataset)

    dataset_mean = np.mean(dataset)

    # Numerator
    numerator = 0
    while t < max_t + 1:
        numerator += (dataset[t - 1] - dataset_mean) * (dataset[t - lags - 1] - dataset_mean)
        t += 1

    t = 1

    # Denominator
    denominator = 0
    while t < max_t + 1:
        denominator += (dataset[t - 1] - dataset_mean) ** 2
        t += 1

    acf_value = numerator / denominator

    return acf_value


def acf_compiler(data, lags):
    """
    :param data: The data used in calculating the ACF values.
    :param lags: The number of lags specified for calculating the ACF values.
    :return: A Numpy array containing the two-sided experimental ACF values.
    """
    acf_values = []

    for k in range(lags + 1):
        acf_values.append(acf_function(data, k))

    acf_values = np.asarray(acf_values)
    acf_values_transform = acf_values[::-1]
    exp_acf_values_ = np.concatenate((np.reshape(acf_values_transform, lags + 1), acf_values[1:]))

    return exp_acf_values_


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
# Part A: Diagnostic tests

# =========================================================
# Confidence intervals
# =========================================================

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

# =========================================================
# Zero/Pole cancellations
# =========================================================

print(100*'-')

# SARIMA (2, 0, 0) x (1, 0, 0, 24)

num = [1, 0]
den = [1, model_1.params[0], model_1.params[1], model_1.params[2]]

zeros = np.roots(num)
poles = np.roots(den)

print('\nZeros of SARIMA (2, 0, 0) x (1, 0, 0, 24):', zeros)
print('Poles of SARIMA (2, 0, 0) x (1, 0, 0, 24):', poles)

# SARIMA (2, 0, 0) x (2, 0, 0, 24)

num = [1, 0]
den = [1, model_2.params[0], model_2.params[1], model_2.params[2], model_2.params[3]]

zeros = np.roots(num)
poles = np.roots(den)

print('\nZeros of SARIMA (2, 0, 0) x (2, 0, 0, 24):', zeros)
print('Poles of SARIMA (2, 0, 0) x (2, 0, 0, 24):', poles)

# SARIMA (1, 1, 0) x (0, 1, 1, 24)

num = [1, model_3.params[1]]
den = [1, model_3.params[0]]

zeros = np.roots(num)
poles = np.roots(den)

print('\nZeros of SARIMA (1, 1, 0) x (0, 1, 1, 24):', zeros)
print('Poles of SARIMA (1, 1, 0) x (0, 1, 1, 24):', poles)

# There are no zero/pole cancellations; thank goodness!

# =========================================================
# Chi-square tests
# =========================================================

print(100*'-')

lags = 55

# SARIMA (2, 0, 0) x (1, 0, 0, 24)

na = 26
nb = 0

osa_predictions_model_1 = []

for i in range(len(Y_train.index)):
    if i == 0:
        osa_predictions_model_1.append(model_1.params[0] * Y_train[i])
    elif 0 < i < 23:
        osa_predictions_model_1.append(model_1.params[0] * Y_train[i] + model_1.params[1] * Y_train[i - 1])
    elif i == 23:
        osa_predictions_model_1.append(model_1.params[0] * Y_train[i] + model_1.params[1] * Y_train[i - 1] + model_1.params[2] * Y_train[i - 23])
    elif i == 24:
        osa_predictions_model_1.append(model_1.params[0] * Y_train[i] + model_1.params[1] * Y_train[i - 1] + model_1.params[2] * Y_train[i - 23] - model_1.params[0] * model_1.params[2] * Y_train[i - 24])
    else:
        osa_predictions_model_1.append(model_1.params[0] * Y_train[i] + model_1.params[1] * Y_train[i - 1] + model_1.params[2] * Y_train[i - 23] - model_1.params[0] * model_1.params[2] * Y_train[i - 24] - model_1.params[1] * model_1.params[2] * Y_train[i - 25])

e = Y_train[1:] - osa_predictions_model_1[:-1]

# re = acf_compiler(data=e, lags=lags)
re = sm.tsa.stattools.acf(e, nlags=lags)

Q = len(Y_train) * np.sum(np.square(re))

lb_value, p_value = sm.stats.acorr_ljungbox(e, lags=lags)

print('The lb value is', lb_value)
print('The p-value is', p_value)

DOF = lags - na - nb

alpha = 0.05

chi_critical = chi2.ppf(1 - alpha, DOF)

if Q < chi_critical:
    print('\nQ value:', Q)
    print('Critical Q:', chi_critical)
    print('The residuals are white.')

else:
    print('\nQ value:', Q)
    print('Critical Q:', chi_critical)
    print('The residuals are not white.')

# # -------------------------------------------------------
# # SARIMA (2, 0, 0) x (2, 0, 0, 24)
# # -------------------------------------------------------

na = 50
nb = 0

osa_predictions_model_2 = []

for i in range(len(Y_train.index)):
    if i == 0:
        osa_predictions_model_2.append(model_2.params[0] * Y_train[i])
    elif 0 < i < 23:
        osa_predictions_model_2.append(model_2.params[0] * Y_train[i] + model_2.params[1] * Y_train[i - 1])
    elif i == 23:
        osa_predictions_model_2.append(model_2.params[0] * Y_train[i] + model_2.params[1] * Y_train[i - 1] + model_2.params[2] * Y_train[i - 23])
    elif i == 24:
        osa_predictions_model_2.append(model_2.params[0] * Y_train[i] + model_2.params[1] * Y_train[i - 1] + model_2.params[2] * Y_train[i - 23] - model_2.params[0] * model_2.params[2] * Y_train[i - 24])
    elif 25 < i < 47:
        osa_predictions_model_2.append(model_2.params[0] * Y_train[i] + model_2.params[1] * Y_train[i - 1] + model_2.params[2] * Y_train[i - 23] - model_2.params[0] * model_2.params[2] * Y_train[i - 24] - model_2.params[1] * model_2.params[2] * Y_train[i - 25])
    elif i == 47:
        osa_predictions_model_2.append(model_2.params[0] * Y_train[i] + model_2.params[1] * Y_train[i - 1] + model_2.params[2] * Y_train[i - 23] - model_2.params[0] * model_2.params[2] * Y_train[i - 24] - model_2.params[1] * model_2.params[2] * Y_train[i - 25] + model_2.params[3] * Y_train[i - 47])
    elif i == 48:
        osa_predictions_model_2.append(model_2.params[0] * Y_train[i] + model_2.params[1] * Y_train[i - 1] + model_2.params[2] * Y_train[i - 23] - model_2.params[0] * model_2.params[2] * Y_train[i - 24] - model_2.params[1] * model_2.params[2] * Y_train[i - 25] + model_2.params[3] * Y_train[i - 47] - model_2.params[0] * model_2.params[3] * Y_train[i - 48])
    else:
        osa_predictions_model_2.append(model_2.params[0] * Y_train[i] + model_2.params[1] * Y_train[i - 1] + model_2.params[2] * Y_train[i - 23] - model_2.params[0] * model_2.params[2] * Y_train[i - 24] - model_2.params[1] * model_2.params[2] * Y_train[i - 25] + model_2.params[3] * Y_train[i - 47] - model_2.params[0] * model_2.params[3] * Y_train[i - 48] - model_2.params[1] * model_2.params[3] * Y_train[i - 49])

e = Y_train[1:] - osa_predictions_model_2[:-1]

# re = acf_compiler(data=e, lags=lags)
re = sm.tsa.stattools.acf(e, nlags=lags)

Q = len(Y_train) * np.sum(np.square(re))

lb_value, p_value = sm.stats.acorr_ljungbox(e, lags=lags)

print('The lb value is', lb_value)
print('The p-value is', p_value)

DOF = lags - na - nb

alpha = 0.05

chi_critical = chi2.ppf(1 - alpha, DOF)

if Q < chi_critical:
    print('\nQ value:', Q)
    print('Critical Q:', chi_critical)
    print('The residuals are white.')

else:
    print('\nQ value:', Q)
    print('Critical Q:', chi_critical)
    print('The residuals are not white.')

# # -------------------------------------------------------
# # SARIMA (1, 1, 0) x (0, 1, 1, 24)
# # -------------------------------------------------------

na = 26
nb = 24

osa_predictions_model_3 = []

for i in range(len(Y_train.index)):
    if i == 0:
        osa_predictions_model_3.append(Y_train[i] + model_3.params[0] * Y_train[i])
    elif 0 < i < 23:
        osa_predictions_model_3.append(Y_train[i] + model_3.params[0] * Y_train[i] - model_3.params[0] * Y_train[i - 1])
    elif i == 23:
        osa_predictions_model_3.append(Y_train[i] + model_3.params[0] * Y_train[i] - model_3.params[0] * Y_train[i - 1] + Y_train[i - 23] + model_3.params[1] * Y_train[i - 23])
    elif i == 24:
        osa_predictions_model_3.append(Y_train[i] + model_3.params[0] * Y_train[i] - model_3.params[0] * Y_train[i - 1] + Y_train[i - 23] + model_3.params[1] * Y_train[i - 23] - Y_train[i - 24] - model_3.params[1] * osa_predictions_model_3[i - 23] - model_3.params[0] * Y_train[24])
    else:
        osa_predictions_model_3.append(Y_train[i] + model_3.params[0] * Y_train[i] - model_3.params[0] * Y_train[i - 1] + Y_train[i - 23] + model_3.params[1] * Y_train[i - 23] - Y_train[i - 24] - model_3.params[1] * osa_predictions_model_3[i - 23] - model_3.params[0] * Y_train[24] + model_3.params[0] * Y_train[i - 25])

e = Y_train[1:] - osa_predictions_model_3[:-1]

# re = acf_compiler(data=e, lags=lags)
re = sm.tsa.stattools.acf(e, nlags=lags)

Q = len(Y_train) * np.sum(np.square(re))

lb_value, p_value = sm.stats.acorr_ljungbox(e, lags=lags)

print('The lb value is', lb_value)
print('The p-value is', p_value)

DOF = lags - na - nb

alpha = 0.05

chi_critical = chi2.ppf(1 - alpha, DOF)

if Q < chi_critical:
    print('\nQ value:', Q)
    print('Critical Q:', chi_critical)
    print('The residuals are white.')

else:
    print('\nQ value:', Q)
    print('Critical Q:', chi_critical)
    print('The residuals are not white.')

# ======================================================================================================================
# Part B: Variance of the error and covariance

# The variance of the error was found be reading the standard errors of the coefficients and squaring them.

# SARIMA (2, 0, 0) x (1, 0, 0, 24)

print('\nSARIMA (2, 0, 0) x (1, 0, 0, 24):\n')

print('Coefficient a1:', np.square(0.003))
print('Coefficient a2:', np.square(0.003))
print('Coefficient a24:', np.square(0.012))

# SARIMA (2, 0, 0) x (2, 0, 0, 24)

print('\nSARIMA (2, 0, 0) x (2, 0, 0, 24):\n')

print('Coefficient a1:', np.square(0.003))
print('Coefficient a2:', np.square(0.003))
print('Coefficient a24:', np.square(0.013))
print('Coefficient a48:', np.square(0.006))

# SARIMA (1, 1, 0) x (0, 1, 1, 24)

print('\nSARIMA (1, 1, 0) x (0, 1, 1, 24):\n')

print('Coefficient a1:', np.square(0.003))
print('Coefficient ma24:', np.square(0.014))

# Covariances of the parameter estimates were found by printing the covariance matrices.

print('\nCovariance matrices')

print('\nSARIMA (2, 0, 0) x (1, 0, 0, 24):\n', model_1.cov_params())

print('\nSARIMA (2, 0, 0) x (2, 0, 0, 24):\n', model_2.cov_params())

print('\nSARIMA (1, 1, 0) x (0, 1, 1, 24):\n', model_3.cov_params())

# ======================================================================================================================
# Part C: Determining whether the model is a biased or unbiased estimator

# After talking with professor Jafari, he said a model is unbiased if the mean of the residuals is non-zero or not
# very close to zero.

model_1_residual_mean = np.mean(osa_predictions_model_1)

model_2_residual_mean = np.mean(osa_predictions_model_2)

model_3_residual_mean = np.mean(osa_predictions_model_3)

print('\nMean of residuals for SARIMA (2, 0, 0) x (1, 0, 0, 24):', model_1_residual_mean)

print('Mean of residuals for SARIMA (2, 0, 0) x (2, 0, 0, 24):', model_2_residual_mean)

print('Mean of residuals for SARIMA (1, 1, 0) x (0, 1, 1, 24):', model_3_residual_mean)

# All three models appear to be biased estimators, but this could be due to errors in calculating one-step ahead
# predictions.

# ======================================================================================================================
# Part D: Check the variance of the residual errors vs the forecast errors

# This part cannot be done without first computing h-step ahead forecasts and comparing the values with the test set to
# compute forecast errors. That step is done below. Upon comparing results, the best model of this project will apply
# h-step ahead predictions using the forecast function, not .forecast().

model_1_residual_var = np.var(osa_predictions_model_1)

model_2_residual_var = np.var(osa_predictions_model_2)

model_3_residual_var = np.var(osa_predictions_model_3)

model_1_forecasts = model_1.forecast(steps=len(y_test))

model_1_forecast_errors = y_test - model_1_forecasts
model_1_forecast_errors_var = np.var(model_1_forecast_errors)

model_2_forecasts = model_2.forecast(steps=len(y_test))
model_2_forecast_errors = y_test - model_2_forecasts
model_2_forecast_errors_var = np.var(model_2_forecast_errors)

model_3_forecasts = model_3.forecast(steps=len(y_test))
model_3_forecast_errors = y_test - model_3_forecasts
model_3_forecast_errors_var = np.var(model_3_forecast_errors)

print('\nVariance of residuals for SARIMA (2, 0, 0) x (1, 0, 0, 24):', model_1_residual_var)
print('Variance of forecast errors for SARIMA (2, 0, 0) x (1, 0, 0, 24):', model_1_forecast_errors_var)

print('\nVariance of residuals for SARIMA (2, 0, 0) x (2, 0, 0, 24):', model_2_residual_var)
print('Variance of forecast errors for SARIMA (2, 0, 0) x (2, 0, 0, 24):', model_2_forecast_errors_var)

print('\nVariance of residuals for SARIMA (1, 1, 0) x (0, 1, 1, 24):', model_3_residual_var)
print('Variance of forecast errors for SARIMA (1, 1, 0) x (0, 1, 1, 24):', model_3_forecast_errors_var)

# ======================================================================================================================
# Part E: If you find that the ARIMA or SARIMA model better represents the dataset, then you can find the model
# accordingly. You are not constraint only to use of ARMA model.

# Because my data is seasonal ARMA (i.e. SARIMA), I decided to test different SARIMA models for this project. Applying
# non-seasonal ARMA models would make no sense, given the seasonal component present in the ACF and PACF plots.

# ======================================================================================================================
# Additional: Plotting the one-step ahead predictions against the training set

# SARIMA (2, 0, 0) x (1, 0, 0, 24)

osa_predictions_model_1 = pd.Series(osa_predictions_model_1, index=Y_train.index)
osa_predictions_model_2 = pd.Series(osa_predictions_model_2, index=Y_train.index)
osa_predictions_model_3 = pd.Series(osa_predictions_model_3, index=Y_train.index)

plt.figure()

plt.plot(Y_train, label='Training data')
plt.plot(y_test, label='Testing data')
plt.plot(osa_predictions_model_1, label='One-step ahead predictions')
plt.plot(model_1_forecasts, label='H-step ahead forecasts')

plt.title('Plot of OSA predictions - SARIMA (2, 0, 0) x (1, 0, 0, 24)')
plt.xlabel('Time (in hours)')
plt.ylabel('PM 2.5 Concentration (µg / cu m)')
plt.legend(loc='upper right')

plt.show()

plt.figure()

plt.plot(Y_train[:50], label='Training data')
plt.plot(osa_predictions_model_1[:50], label='One-step ahead predictions')

plt.title('OSA predictions - SARIMA (2, 0, 0) x (1, 0, 0, 24) (first 50 samples)')
plt.xlabel('Time (in hours)')
plt.ylabel('PM 2.5 Concentration (µg / cu m)')
plt.legend(loc='upper right')

plt.xticks(rotation=45)

plt.show()

# SARIMA (2, 0, 0) x (2, 0, 0, 24)

plt.figure()

plt.plot(Y_train, label='Training data')
plt.plot(y_test, label='Testing data')
plt.plot(osa_predictions_model_2, label='One-step ahead predictions')
plt.plot(model_2_forecasts, label='H-step ahead forecasts')

plt.title('Plot of OSA predictions - SARIMA (2, 0, 0) x (2, 0, 0, 24)')
plt.xlabel('Time (in hours)')
plt.ylabel('PM 2.5 Concentration (µg / cu m)')
plt.legend(loc='upper right')

plt.show()

plt.figure()

plt.plot(Y_train[:50], label='Training data')
plt.plot(osa_predictions_model_2[:50], label='One-step ahead predictions')

plt.title('OSA predictions - SARIMA (2, 0, 0) x (2, 0, 0, 24) (first 50 samples)')
plt.xlabel('Time (in hours)')
plt.ylabel('PM 2.5 Concentration (µg / cu m)')
plt.legend(loc='upper right')

plt.xticks(rotation=45)

plt.show()

# SARIMA (1, 1, 0) x (0, 1, 1, 24)

plt.figure()

plt.plot(Y_train, label='Training data')
plt.plot(y_test, label='Testing data')
plt.plot(osa_predictions_model_3, label='One-step ahead predictions')
plt.plot(model_3_forecasts, label='H-step ahead forecasts')

plt.title('Plot of OSA predictions - SARIMA (1, 1, 0) x (0, 1, 1, 24)')
plt.xlabel('Time (in hours)')
plt.ylabel('PM 2.5 Concentration (µg / cu m)')
plt.legend(loc='upper right')

plt.show()

plt.figure()

plt.plot(Y_train[:50], label='Training data')
plt.plot(osa_predictions_model_3[:50], label='One-step ahead predictions')

plt.title('OSA predictions - SARIMA (1, 1, 0) x (0, 1, 1, 24) (first 50 samples)')
plt.xlabel('Time (in hours)')
plt.ylabel('PM 2.5 Concentration (µg / cu m)')
plt.legend(loc='upper right')

plt.xticks(rotation=45)

plt.show()
