import pandas as pd
from pathlib import Path
from datetime import date, datetime, timedelta
import time
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score,mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

home_price_wt = 0.5
weather_wt = 0.3
crime_wt = 0.1
sq_ft_wt = 0.1

training_data = pd.read_csv("Suitability_score_house.csv")

lin_reg = LinearRegression()

X = training_data.drop(["status", "city", "brokered_by", "zip_code", 'prev_sold_date'], axis='columns')
Y = training_data["Suitability"]
train_x, test_x, train_y, test_y = train_test_split(X,Y,test_size=0.3, random_state=42, shuffle=True)

lin_reg.fit(train_x, train_y)
y_pred_lin = lin_reg.predict(test_x)

root_mse = root_mean_squared_error(test_y, y_pred_lin)
r2_lin_pred = r2_score(test_y, y_pred_lin)
print("Score report:")
print("Root MSE: ", root_mse)
print("R-squared: ", r2_lin_pred)