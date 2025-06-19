# -*- coding: utf-8 -*-
"""
Created on Sat May  4 19:48:37 2024

@author: ertugrulkirac
"""

import numpy as np
import pandas as pd

boston = pd.read_csv('boston.csv')

print(boston.head())

boston.info()


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

import warnings
warnings.filterwarnings('ignore')

data = boston.drop(['MEDV'], axis=1)


for i in data.columns:
    data[i] = data[i].fillna(data[i].mean())
print(data)

for i in data.columns:
    data[i] = scaler.fit_transform(data[[i]]) 
print(data)

from sklearn.model_selection import train_test_split

x = data
y = boston['MEDV']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=0)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


print('----LinearRegression-------')

lr = LinearRegression()
lr.fit(x_train, y_train)

y_pred_lr = lr.predict(x_test)

"""print('mse ', mean_squared_error(y_test, y_pred_lr))"""
print('r2_score',r2_score(y_test, y_pred_lr) )


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

print('----RandomForestRegressor------')


rnd_reg = RandomForestRegressor(n_estimators=100)
rnd_reg.fit(x_train, y_train)

y_pred_rnd_reg = rnd_reg.predict(x_test)

"""print('mse ', mean_squared_error(y_test, y_pred_rnd_reg))"""
print('r2_score',r2_score(y_test, y_pred_rnd_reg) )


print('------GradientBoostingRegressor-----')


gb = GradientBoostingRegressor()
gb.fit(x_train, y_train)

y_pred_gb = gb.predict(x_test)

"""print('mse ', mean_squared_error(y_test, y_pred_gb))"""
print('r2_score',r2_score(y_test, y_pred_gb) )

#pip install xgboost

import xgboost as xgb


print('---XGBRegressor----')
xg = xgb.XGBRegressor()
xg.fit(x_train, y_train)

y_pred_xg = xg.predict(x_test)

"""print('mse ', mean_squared_error(y_test, y_pred_xg))"""
print('r2_score',r2_score(y_test, y_pred_xg) )

from sklearn.linear_model import Lasso


print('---Lasso----')

lss = Lasso(alpha=0.3, fit_intercept=True, max_iter=100)
lss.fit(x_train, y_train)

y_pred_lss = lss.predict(x_test)

"""print('mse ', mean_squared_error(y_test, y_pred_lss))"""
print('r2_score',r2_score(y_test, y_pred_lss) )




