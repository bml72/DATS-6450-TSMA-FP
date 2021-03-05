import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf

# ======================================================================================================================
# Final Project: Multiple linear regression
# Benjamin Lee
# Data Science 6450-15: Time Series Modeling & Analysis
# 16 December 2020
# ======================================================================================================================
# Applicable functions


def acf_function(dataset, _lags_):
    """
    :param dataset: The data used in calculating the ACF values.
    :param _lags_: The number of lags specified for calculating the ACF values.
    :return: The ACF values associated with the given dataset at a specified lag value.
    # Values for t are adjusted to account for the indexing differences between Python and the handwritten ACF.
    """
    t = _lags_ + 1

    max_t = len(dataset)

    dataset_mean = np.mean(dataset)

    # Numerator
    numerator = 0
    while t < max_t + 1:
        numerator += (dataset[t - 1] - dataset_mean) * (dataset[t - _lags_ - 1] - dataset_mean)
        t += 1

    t = 1

    # Denominator
    denominator = 0
    while t < max_t + 1:
        denominator += (dataset[t - 1] - dataset_mean) ** 2
        t += 1

    acf_value = numerator / denominator

    return acf_value


def one_sided_acf_compiler(data, _lags_):
    """
    :param data: The data used in calculating the ACF values.
    :param _lags_: The number of lags specified for calculating the ACF values.
    :return: A Numpy array containing the two-sided experimental ACF values.
    """
    acf_values = []

    for k in range(_lags_ + 1):
        acf_values.append(acf_function(data, k))

    acf_values = np.asarray(acf_values)

    return acf_values


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

# Separating the data into predictor and response; during office hours, it was recommended to remove the encoded
# categorical variable from feature selection.
predictors = df.drop(columns=['pm2.5', 'cbwd'])

predictors = sm.add_constant(predictors)

response = pm.copy(deep=True)

# Splitting the predictor and response into training and testing sets.
X_train, x_test, Y_train, y_test = train_test_split(predictors, response, shuffle=False, test_size=0.2)

# ======================================================================================================================
# Analysis of 'q9_feature_selection.py'

# In 'q9_feature_selection.py,' forward and backward stepwise regression resulted in selecting all features for the
# multiple linear regression model.

# ======================================================================================================================
# Part A: Creating the model and performing one-step ahead prediction with comparison against the test set

# NOTE: I am a bit confused by the prompt of this question. As I've come to understand from previous assignments,
# one-step ahead predictions are performed on and compared with the training set. Additionally, in part D the ACF
# values computed are for the residuals, which are based on the difference between one-step ahead predictions and the
# original training set values. Thus, I believe the prompt should say "compare the performance versus the test set."

model = sm.OLS(Y_train, X_train).fit()

osa_predictions = model.predict(X_train)

figure, axes = plt.subplots()

axes.plot(Y_train, label='Training Dataset')
axes.plot(y_test, label='Testing Dataset')
axes.plot(osa_predictions, label='One-step ahead predictions')

plt.xlabel('Observations')
plt.ylabel('PM 2.5 concentration (µg/cu m)')
plt.title('Multiple Linear Regression Model: One-Step Ahead Predictions')
plt.legend(loc='upper right')

plt.show()

# Analysis: The one-step ahead predictions cover the bulk of the training set, but values above a PM 2.5 concentration
# of ~ 200 are left uncovered. These predictions are okay, but do not cover cases where the PM 2.5 concentration spikes.
# Out of curiosity, multi-step ahead forecasts were constructed and plotted below. The multi-step ahead forecasts were
# compared with the testing set.

model = sm.OLS(Y_train, X_train).fit()

msa_predictions = model.predict(x_test)

figure, axes = plt.subplots()

axes.plot(Y_train, label='Training Dataset')
axes.plot(y_test, label='Testing Dataset')
axes.plot(msa_predictions, label='Multi-step ahead predictions')

plt.xlabel('Observations')
plt.ylabel('PM 2.5 concentration (µg/cu m)')
plt.title('Multiple Linear Regression Model: Multi-Step Ahead Predictions')
plt.legend(loc='upper right', prop={'size': 9})

plt.show()

# Analysis: The results of multi-step ahead predictions are similar to the one-step ahead predictions. The multi-step
# ahead predictions cover the bulk of the testing set, but original data spikes above ~ 200 are left uncovered. The same
# process is repeated with the first and last months of data to gain a better handle on how the one-step ahead
# predictions and multi-step ahead forecasts compare to the training and testing sets, respectively.

df_first_month = df.loc[:'2010-02-01']
pm_first_month = pm.loc[:'2010-02-01']

predictors_first_month = df_first_month.drop(columns='pm2.5')

predictors_first_month = sm.add_constant(predictors_first_month)

response_first_month = pm_first_month.copy(deep=True)

X_train_first_month, x_test_first_month, Y_train_first_month, y_test_first_month = train_test_split(
    predictors_first_month, response_first_month, shuffle=False, test_size=0.2)

# Model building and plotting: one-step ahead predictions

model_first_month = sm.OLS(Y_train_first_month, X_train_first_month).fit()

osa_predictions_first_month = model_first_month.predict(X_train_first_month)

figure, axes = plt.subplots()

axes.plot(Y_train_first_month, label='Training Dataset')
axes.plot(y_test_first_month, label='Testing Dataset')
axes.plot(osa_predictions_first_month, label='One-step ahead predictions')

plt.xlabel('Observations')
plt.ylabel('PM 2.5 concentration (µg/cu m)')
plt.title('Multiple Linear Regression Model: One-Step Predictions (Jan \'10)')
plt.legend(loc='upper right')

plt.xticks(rotation=45)

plt.show()

# Model building and plotting: multi-step ahead predictions

model_first_month_multi = sm.OLS(Y_train_first_month, X_train_first_month).fit()

msa_predictions_first_month = model_first_month_multi.predict(x_test_first_month)

figure, axes = plt.subplots()

axes.plot(Y_train_first_month, label='Training Dataset')
axes.plot(y_test_first_month, label='Testing Dataset')
axes.plot(msa_predictions_first_month, label='Multi-step ahead predictions')

plt.xlabel('Observations')
plt.ylabel('PM 2.5 concentration (µg/cu m)')
plt.title('Multiple Linear Regression Model: Multi-Step Predictions (Jan \'10)')
plt.legend(loc='upper right', prop={'size': 9})

plt.xticks(rotation=45)

plt.show()

# ======================================================================================================================
# Part B: Hypothesis tests

print(model.summary())

# The p-value of the F-statistic is 0.00. Because the p-value of the F-test is below the chosen threshold (0.05), we
# reject the null hypothesis and conclude that the model provides a better fit than the intercept-only model. The
# results of the t-tests for the coefficients of the predictors all display a p-value of 0.000. Because the p-value of
# the t-tests are below the chosen threshold of 5% (0.05), we reject the null hypothesis for all cases and conclude that
# all of the coefficients are not equal to zero.

# ======================================================================================================================
# Part C: AIC, BIC, RMSE, R-squared and adjusted R-squared values

# The summary printed in part B displays all relevant statistics, save for the RSME. The AIC and BIC are very large
# (79,840 and 79,890, respectively), which are not very good; because the AIC and BIC values are directly proportional
# to the SSE, this means the SSE is also very high. If the SSE is very high, then there is a large degree of error in
# this multiple linear regression model. Similarly, the R-squared and adjusted R-squared values are very small (0.239
# and 0.238, respectively). Because the R-squared and adjusted R-squared values are so small, only a small proportion
# (23.8% for R-squared and 23.9% for adjusted R-squared) of the dependent variable (PM 2.5 concentration) can be
# reasonably explained by an independent variable of the model.

rs_me_osa = np.sqrt(mean_squared_error(Y_train, osa_predictions))

print('\nThe root mean squared error for one-step ahead predictions is', rs_me_osa)

rs_me_msa = np.sqrt(mean_squared_error(y_test, msa_predictions))

print('The root mean squared error for multi-step ahead predictions is', rs_me_msa)

# The root mean squared errors of the one-step and multi-step ahead predictions are relatively large (71.96915 and
# 96.74490, respectively). The large values of the root mean squared error suggest that there exists a large deviation
# of the residuals and forecast errors. This means that the data is not very concentrated around the line of best fit
# for the residuals and forecast errors.

# ======================================================================================================================
# Part D: ACF of the residuals

residuals = []

for val in range(len(osa_predictions)):
    residuals.append(Y_train[val] - osa_predictions[val])

lags = 50

plt.figure()

plot_acf(residuals, lags=lags)

plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('ACF of The Residuals with {} Lags'.format(lags))

plt.show()

# Analysis: The patterns in the ACF plot of the residuals follows the pattern of the ACF plot for the raw dataset. There
# exist slight spikes or humps in values roughly every 24 lags. This is indicative of a seasonal pattern of 24, which
# will be implemented in SARIMA modelling later on.

# ======================================================================================================================
# Part E: Q-value of residuals

residuals = np.asarray(residuals)

acf_values = one_sided_acf_compiler(residuals, lags)

summation = 0

# After receiving my grade for Homework 7, the TA noted that "Q-value have to calculate start from 1." I interpreted
# this to mean the Q-value should be calculated from lag 1 onwards; this should not include lag 0. Therefore,
# acf_values[1:] accounts for this issue.
for val in acf_values[1:]:
    summation += val ** 2

Q = len(Y_train) * np.sum(np.square(acf_values[1:]))

print('\nThe Q value of the residuals is', Q)

# The ACF plot of the residuals denotes a lack of swift decay and displays all ACF values above the insignificance
# region (lags = 50). Because the ACF values are not close to zero, the Q value is proportionately large.

# ======================================================================================================================
# Part F: Variance and mean of the residuals

res_mean = np.mean(residuals)

print('\nThe mean of the residuals is', res_mean)

res_var = np.var(residuals)

print('The variance of the residuals is', res_var)

# The mean of the residuals is near-zero, and the variance of the residuals is extremely large. The variance follows the
# pattern of the RSME, whose large value suggested a proportionately large deviation from the line of best fit for the
# residuals.

# ======================================================================================================================
# Overall analysis of multiple linear regression

# Overall, multiple linear regression is a very bad model for this dataset. The R-squared value and adjusted R-squared
# values are very low, and the AIC/BIC values are very large. This means that the change in PM 2.5 concentration cannot
# be accurately explained by changes in the other features of the dataset, and the SSE of this data is very large.
