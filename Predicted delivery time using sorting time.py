# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 14:20:22 2022

@author: Sai pranay
"""
#-----------------------IMPORTING_THE_DATA_SET---------------------------------

import pandas as pd
import numpy as np
dt = pd.read_csv("E:\\DATA_SCIENCE_ASS\\SIMPLR_LINEAR_REGRESSION\\delivery_time.csv")
print(dt)
dt.shape
list(dt)
dt.ndim
dt.describe().T
dt.info()
dt.hist()
#-----------------------------SPLITTING_THE_DATA_SET---------------------------


x = dt['Sorting Time']
print(x)
x.ndim
x.shape


y = dt['Delivery Time']
print(y)
y.ndim
y.shape

#----------------------changing_the_dimension----------------------------------

import numpy as np
x = x[:, np.newaxis]
x.shape
x.ndim

#----------------------model_deploymenet---------------------------------------


from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(x,y)
model.intercept_
model.coef_


Y_Pred = model.predict(x)
Y_Pred


#---------------------------------scatter plot---------------------------------

dt.plot.scatter(x='Sorting Time', y='Delivery Time')

#-------------------------checking_the_correlation_points----------------------

dt.corr()


#------------------------------ Plot_outputs-----------------------------------
import matplotlib.pyplot as plt
plt.scatter(x, y,  color='black')
plt.plot(x, Y_Pred, color='red')
plt.show()



Y_error = y-Y_Pred
print(Y_error)

#-------------------------importing_the_mean_squared_error_--------------------

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y,Y_Pred)
mse

RMSE = np.sqrt(mse)
RMSE

#--------------------square_root_transformation--------------------------------

x_sqrt = np.sqrt(dt['Sorting Time'])
x_sqrt
x_sqrt.ndim

y_sqrt = np.sqrt(dt['Delivery Time'])
y_sqrt
y_sqrt.ndim

x_sqrt=x_sqrt[:,np.newaxis]
x_sqrt.ndim

y_sqrt=y_sqrt[:,np.newaxis]
y_sqrt.ndim


from sklearn.linear_model import LinearRegression
mode4 = LinearRegression().fit(x_sqrt,y_sqrt)
mode4.intercept_
mode4.coef_


Y_Pred4 = mode4.predict(x_sqrt)
Y_Pred4

Y_error = y_sqrt-Y_Pred4
print(Y_error)


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_sqrt,Y_Pred4)
mse

RMSE = np.sqrt(mse)
RMSE

import statsmodels.api as sm


model_2 = sm.OLS(y_sqrt, x_sqrt).fit()
predictions = model_2.predict(x_sqrt)
model_2.summary()

#--------------------------LOG_TRANSFORMATION-----------------------------------

y_log = np.log(dt['Delivery Time'])
x_log = np.log(dt['Sorting Time'])

x_log=x_log[:,np.newaxis]
x_log.ndim

y_log=y_log[:,np.newaxis]
y_log.ndim


from sklearn.linear_model import LinearRegression
mode5 = LinearRegression().fit(x_log,y_log)
mode5.intercept_
mode5.coef_


Y_Pred5 = mode5.predict(x_log)
Y_Pred5

Y_error = y_log-Y_Pred5
print(Y_error)


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_log,Y_Pred5)
mse

RMSE = np.sqrt(mse)
RMSE
