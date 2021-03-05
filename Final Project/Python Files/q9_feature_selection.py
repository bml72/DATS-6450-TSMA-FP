import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

# ======================================================================================================================
# Final Project: Feature selection
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

# Separating the data into predictor and response; during office hours, it was recommended to remove the encoded
# categorical variable from feature selection.
predictors = df.drop(columns=['pm2.5', 'cbwd'])

predictors = sm.add_constant(predictors)

response = pm.copy(deep=True)

# Splitting the predictor and response into training and testing sets.
X_train, x_test, Y_train, y_test = train_test_split(predictors, response, shuffle=False, test_size=0.2)

# ======================================================================================================================
# Forward stepwise regression

X_train_forward = X_train.copy(deep=True)

for val in range(len(X_train_forward.columns)):
    forward_model = sm.OLS(Y_train, X_train_forward[['const', X_train_forward.columns[val]]]).fit()
    # print(forward_model.summary())

# DEWP:     t-val = 27.135  |   p-val = 0.000
# TEMP:     t-val = 9.494   |   p-val = 0.000
# PRES:     t-val = -21.489 |   p-val = 0.000
# Iws:      t-val = -20.924 |   p-val = 0.000
# Is:       t-val = -2.033  |   p-val = 0.042
# Ir:       t-val = -4.569  |   p-val = 0.000

# Between the variables with the smallest p-val, dew point has the largest t-val and is put into the basket.

for val in range(len(X_train_forward.columns)):
    forward_model = sm.OLS(Y_train, X_train_forward[['const', 'DEWP', X_train_forward.columns[val]]]).fit()
    # print(forward_model.summary())

# TEMP:     t-val = -26.454 |   p-val = 0.000
# PRES:     t-val = -1.570  |   p-val = 0.117
# Iws:      t-val = -14.649 |   p-val = 0.000
# Is:       t-val = -0.281  |   p-val = 0.779
# Ir:       t-val = -7.863  |   p-val = 0.000

# Between the variables with the smallest p-val, cumulative hours of rain has the largest t-val and is put into the
# basket.

for val in range(len(X_train_forward.columns)):
    forward_model = sm.OLS(Y_train, X_train_forward[['const', 'DEWP', 'Ir', X_train_forward.columns[val]]]).fit()
    # print(forward_model.summary())

# TEMP:     t-val = -28.078 |   p-val = 0.000
# PRES:     t-val = -0.983  |   p-val = 0.325
# Iws:      t-val = -14.779 |   p-val = 0.000
# Is:       t-val = -0.327  |   p-val = 0.744

# Between the variables with the smallest p-val, cumulative wind speed has the largest t-val and is put into the basket.

for val in range(len(X_train_forward.columns)):
    forward_model = sm.OLS(Y_train, X_train_forward[['const', 'DEWP', 'Ir', 'Iws', X_train_forward.columns[val]]]).fit()
    # print(forward_model.summary())

# TEMP:     t-val = -25.512 |   p-val = 0.000
# PRES:     t-val = -2.814  |   p-val = 0.005
# Is:       t-val = 0.103   |   p-val = 0.918

# Temperature possessed the smallest p-value below the chosen threshold (0.005) and is put into the basket.

for val in range(len(X_train_forward.columns)):
    forward_model = sm.OLS(Y_train, X_train_forward[['const', 'DEWP', 'Ir', 'Iws', 'TEMP', X_train_forward.columns[val]]]).fit()
    # print(forward_model.summary())

# PRES:     t-val = -17.240 |   p-val = 0.000
# Is:       t-val = -9.540  |   p-val = 0.000

# Between the variables with the smallest p-val, cumulative hours of snow has the largest t-val and is put into the
# basket.

for val in range(len(X_train_forward.columns)):
    forward_model = sm.OLS(Y_train, X_train_forward[['const', 'DEWP', 'Ir', 'Iws', 'TEMP', 'Is', X_train_forward.columns[val]]]).fit()
    # print(forward_model.summary())

# PRES:     t-val = -17.390 |   p-val = 0.000

# The p-value for pressure is below the chosen threshold (0.05) and is put into the basket.

forward_model = sm.OLS(Y_train, X_train_forward[['const', 'DEWP', 'Ir', 'Iws', 'TEMP', 'Is', 'PRES']]).fit()
print(forward_model.summary())

# ======================================================================================================================
# Analysis of forward stepwise regression

# The results of forward stepwise regression conclude that all features of the dataset are significant. At each step of
# the process, the feature included in the model possessed a p-value below the chosen threshold (0.05). The OLS
# regression results display a R-squared value of 0.239 and an adjusted R-squared value of 0.238, which are very low.
# The results of the r-squared values suggest that only 23.8%/23.9% of the dependent variable (PM 2.5 concentration) can
# be reasonably explained by an independent variable (the features included in forward stepwise regression). This aligns
# with the results of the correlation matrix produced in 'q5_description.py,' in which none of the features possessed a
# strong correlation with the dependent variable (none above ~ 0.3).

# ======================================================================================================================
# Backward stepwise regression

X_train_backward = X_train.copy(deep=True)

backward_model = sm.OLS(Y_train, X_train_backward).fit()
print(backward_model.summary())

# The p-values of all features are below the chosen threshold (0.05); none of the features should be removed.

# ======================================================================================================================
# Analysis of backward stepwise regression

# Because none of the features ended up being removed from forward stepwise regression, the results of backward stepwise
# regression are the same, save for the order of the features in the printout.
