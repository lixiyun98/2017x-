# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 13:52:38 2017

@author: LZDSLI
"""
import pandas as pd
import math
from sklearn.metrics import mean_squared_error
data_path = 'I:/day/test_day.csv'
rides = pd.read_csv(data_path)
rides.head()
data_path = 'I:/day/day_out.csv'
rodes=pd.read_csv(data_path)
rodes.head()
mse=mean_squared_error(rodes['cnt'],rides['cnt'])
print("RMSE using:    ",end='')
rmse=math.sqrt(mse)
print(rmse,'\n')
data_path = 'I:/hour/test_hour.csv'
rides = pd.read_csv(data_path)
rides.head()
data_path = 'I:/hour/hour_out.csv'
rodes=pd.read_csv(data_path)
rodes.head()
mse=mean_squared_error(rodes['cnt'],rides['cnt'])
print("RMSE using:    ",end='')
rmse=math.sqrt(mse)
print(rmse,'\n')