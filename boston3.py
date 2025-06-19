# -*- coding: utf-8 -*-
"""
Created on Sat May  4 19:48:37 2024
@author: ertugrulkirac
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor
import xgboost as xgb

# Veriyi oku
boston = pd.read_csv('boston.csv')
print("İlk 5 Satır:\n", boston.head())
boston.info()

# Hedef değişkeni ayır
X = boston.drop('MEDV', axis=1)
y = boston['MEDV']

# Eksik verileri ortalama ile doldur
for i in X.columns:
    X[i] = X[i].fillna(X[i].mean())

# Ölçekleme
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Veriyi ayır
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.7, random_state=10)

# Model değerlendirme fonksiyonu
def evaluate_model(model, name):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"--- {name} ---")
    print("R² Score:", round(r2, 4))
    print("MSE     :", round(mse, 4))
    print("-" * 40)
    return {'Model': name, 'R2 Score': r2, 'MSE': mse}

# Modellerin değerlendirilmesi
results = []
results.append(evaluate_model(LinearRegression(), "LinearRegression"))
results.append(evaluate_model(BaggingRegressor(n_estimators=300), "BaggingRegressor"))
results.append(evaluate_model(RandomForestRegressor(n_estimators=100), "RandomForestRegressor"))
results.append(evaluate_model(GradientBoostingRegressor(), "GradientBoostingRegressor"))
results.append(evaluate_model(xgb.XGBRegressor(), "XGBRegressor"))
results.append(evaluate_model(Lasso(alpha=0.3), "Lasso"))
results.append(evaluate_model(AdaBoostRegressor(n_estimators=50), "AdaBoostRegressor"))


# Sonuçları tablo halinde göster
results_df = pd.DataFrame(results)
print("\n--- Karşılaştırma Tablosu ---")
print(results_df.sort_values(by="R2 Score", ascending=False).reset_index(drop=True))
