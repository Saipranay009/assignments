# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 14:50:35 2022

@author: Sai pranay
"""
#-----------------------IMPORTING_THE_DATA_SET---------------------------------

import pandas as pd
sd = pd.read_csv("E:\DATA_SCIENCE_ASS\SIMPLR_LINEAR_REGRESSION\\Salary_Data.csv")
sd
sd.shape
list(sd)
sd.ndim

#-----------------------------SPLITTING_THE_DATA_SET---------------------------


x = sd['YearsExperience']
x
x.ndim
x.shape


y = sd['Salary']
y
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


#------------------------------scatter_plot------------------------------------

sd.plot.scatter(x='YearsExperience', y='Salary')


#-------------------------checking_the_correlation_points----------------------


sd.corr()



#-----------------------------Plot_outputs-------------------------------------
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


#------------------------------------------------------------------------------

import statsmodels.api as sm


model_1 = sm.OLS(y, x).fit()
predictions = model_1.predict(x)
model_1.summary()



#--------------------square_root_transformation--------------------------------

x_sqrt = np.sqrt(sd['YearsExperience'])
x_sqrt
x_sqrt.ndim

y_sqrt = np.sqrt(sd['Salary'])
y_sqrt
y_sqrt.ndim


x_sqrt = np.sqrt(sd['YearsExperience'])
x_sqrt
x_sqrt.ndim

y_sqrt = np.sqrt(sd['Salary'])
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


model_2 = sm.OLS(y_sqrt, x_sqrt).fit()
predictions = model_2.predict(x_sqrt)
model_2.summary()

#-------------------------LOG_TRANSFORMATIOM-----------------------------------

y_log = np.log(sd['Salary'])
x_log = np.log(sd['YearsExperience'])


from sklearn.linear_model import LinearRegression
mode5 = LinearRegression().fit(x_log,y_log)
mode5.intercept_
mode5.coef_

x_log=x_log[:,np.newaxis]

x_log.ndim

y_log=y_log[:,np.newaxis]


y_log.ndim



Y_Pred5 = mode5.predict(x_log)
Y_Pred5

Y_error = y-Y_Pred5
print(Y_error)


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_log,Y_Pred5)
mse

RMSE = np.sqrt(mse)
RMSE