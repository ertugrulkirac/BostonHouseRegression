# -*- coding: utf-8 -*-
"""
Created on Sat May  4 10:42:10 2024

@author: ertug
"""

import numpy as np # linear algebra
import pandas as pd # data processing

boston = pd.read_csv('boston.csv')

print(boston.head()) #Return the first 5 rows of the DataFrame.

boston.info()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import warnings
warnings.filterwarnings('ignore')
data = boston.drop(['MEDV'], axis = 1)
for i in data.columns:
    data[i] = scaler.fit_transform(data[[i]])
    
print(data)

from sklearn.model_selection import train_test_split
x = data
y = boston['MEDV']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

print('------------------LinearRegression---------------------')

lr = LinearRegression()
lr.fit(x_train, y_train)
print('training ', lr.score(x_train, y_train))
print('testing ', lr.score(x_test, y_test))
y_pred_lr = lr.predict(x_test)
print('mse ', mean_squared_error(y_test, y_pred_lr))
print('rmse', np.sqrt(mean_squared_error(y_test, y_pred_lr)))

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb


print('------------------RandomForestRegressor---------------------')

rfr = RandomForestRegressor(n_estimators=100)
rfr.fit(x_train, y_train)
y_pred_rfc = rfr.predict(x_test)
print('mse ', mean_squared_error(y_test, y_pred_rfc))
print('rmse', np.sqrt(mean_squared_error(y_test, y_pred_rfc)))
print('r2 ', r2_score(y_test, y_pred_rfc))

print('------------------GradientBoostingRegressor---------------------')

gbr = GradientBoostingRegressor()
gbr.fit(x_train, y_train)
y_pred_gbr = gbr.predict(x_test)

print('mse ', mean_squared_error(y_test, y_pred_gbr))
print('rmse', np.sqrt(mean_squared_error(y_test, y_pred_gbr)))
print('r2 ', r2_score(y_test, y_pred_gbr))

xg = xgb.XGBRegressor()
xg.fit(x_train,y_train)
y_pred_xg = xg.predict(x_test)

print('------------------XGBRegressor---------------------')

print('mse ', mean_squared_error(y_test, y_pred_xg))
print('rmse', np.sqrt(mean_squared_error(y_test, y_pred_xg)))
print('r2 ', r2_score(y_test, y_pred_xg))