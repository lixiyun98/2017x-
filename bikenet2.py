# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 22:15:30 2017

@author: LZDSLI
"""
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor    
from sklearn.cross_validation import train_test_split
import sklearn.cross_validation
from sklearn import feature_selection
from sklearn.svm import SVR
import numpy 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
import pylab as pl
import math
import seaborn as sns

data_path = 'I:/day/train_day.csv'
rides = pd.read_csv(data_path)
rides.head()
data_path = 'I:/day/day_out.csv'
rodes=pd.read_csv(data_path)
#cnt_correlations=rides.corr()['cnt']
#print("\n Reading success! cnt-correlations：\n")
#print(cnt_correlations)
rides[:24*10].plot(x='dteday', y='cnt')
dummy_fields = ['season', 'weathersit', 'mnth','weekday','yr']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)
    dummies = pd.get_dummies(rodes[each], prefix=each, drop_first=False)
    rodes = pd.concat([rodes, dummies], axis=1)
cnt_correlations=rides.corr()['cnt']
print("\n Reading success! cnt-correlations：\n")
print(cnt_correlations)
#quant_features = [ 'temp']
## Store scalings in a dictionary so we can convert back later
#scaled_features = {}
#for each in quant_features:
#    mean, std = rides[each].mean(), rides[each].std()
#    scaled_features[each] = [mean, std]
#    rides.loc[:, each] = (rides[each] - mean)/std
data=rides
tesdata=rodes
#fig,axes = plt.subplots(nrows=3,ncols=2,figsize=(15,20))
#sns.boxplot(y="cnt",data=data,ax=axes[0][0])
##sns.boxplot(x="hr",y="cnt",data=data,ax=axes[0][1])
#sns.boxplot(x="weatherist",y="cnt",data=data,ax=axes[1][0])
#sns.boxplot(x="weekday",y="cnt",data=data,ax=axes[1][1],order=["Monday","Tuesday"
#                                                                ,"Wednesday","Thursday",
#                                                                "Friday","Saturday","Sunday"])
#sns.boxplot(x="season",y="count",data=data,ax=axes[2][0])
#sns.boxplot(x="workingday",y="count",data=data,ax=axes[2][1])
fields_to_drop = ['ind', 'weekday','dteday']
data = rides.drop(fields_to_drop, axis=1)
tesdata= rodes.drop(fields_to_drop, axis=1)
data.head()
tesdata.head()
columns=list(tesdata.columns)
columns.remove('cnt')
columns.remove('casual')
columns.remove('registered')
columns.remove('holiday')
columns.remove('weathersit_2')
columns.remove('yr')

#data,tesdata=train_test_split(data[columns],tesdata['cnt'],test_size=0.4,random_state=0)
#columns.remove('registered')
#columns=['yr','temp','windspeed','season_1','season_2','season_3','weathersit_1','weathersit_2','weathersit_3','workingday']
train=data
test=tesdata
#sX=StandardScaler()
#sY=StandardScaler()
#train=sX.fit_transform(train)
#test=sY.fit_transform(test)
#model=LinearRegression()
#model.fit(train[columns],train['casual'])
#predictions1=model.predict(test[columns])
#model.fit(train[columns],train['registered'])
#predictions2=model.predict(test[columns])
#predictions=predictions1+predictions2
#mse=mean_squared_error(test['cnt'],predictions)
#print("RMSE using LinearRegression:    ",end='')
#rmse=math.sqrt(mse)
#print(rmse,'\n')
#sgdr=SGDRegressor()
#model.fit(train[columns],train['casual'])
#predictions1=model.predict(test[columns])
#model.fit(train[columns],train['registered'])
#predictions2=model.predict(test[columns])
#predictions=predictions1+predictions2
#mse=mean_squared_error(test['cnt'],predictions)
#print("RMSE using SGDRegression:    ",end='')
#rmse=math.sqrt(mse)
#print(rmse,'\n')
#linear_svr=SVR(kernel='linear')
#linear_svr.fit(train[columns],train['cnt'])
#predictions=linear_svr.predict(test[columns])
#mse=mean_squared_error(test['cnt'],predictions)
#print("MSE using SVRlinear:    ",end='')
#rmse=math.sqrt(mse)
#print(rmse,'\n')
#poly_svr=SVR(kernel='poly')
#poly_svr.fit(train[columns],train['cnt'])
#predictions=poly_svr.predict(test[columns])
#print("MSE using SVRpoly:    ",end='')
#rmse=math.sqrt(mse)
#print(rmse,'\n')
#rbf_svr=SVR(kernel='rbf')
#rbf_svr.fit(train[columns],train['cnt'])
#predictions=rbf_svr.predict(test[columns])
#print("RMSE using SVRrbf:    ",end='')
#rmse=math.sqrt(mse)
#print(rmse,'\n')
#model=DecisionTreeRegressor(min_samples_leaf=5)
#model.fit(train[columns],train['casual'])
#predictions1=model.predict(test[columns])
#model.fit(train[columns],train['registered'])
#predictions2=model.predict(test[columns])
#predictions=predictions1+predictions2
#mse=mean_squared_error(test['cnt'],predictions)
#print("RMSE using DecisionTreeRegressor:    ",end='')
#rmse=math.sqrt(mse)
#print(rmse,'\n')

model=RandomForestRegressor()
model.fit(train[columns],train['casual'])
predictions1=model.predict(test[columns])
model.fit(train[columns],train['registered'])
predictions2=model.predict(test[columns])
predictions=predictions1+predictions2
mse=mean_squared_error(test['cnt'],predictions)
print("RMSE using RandomForestRegressor:    ",end='')
rmse=math.sqrt(mse)
print(rmse,'\n')
model=ExtraTreesRegressor()
model.fit(train[columns],train['casual'])
predictions1=model.predict(test[columns])
model.fit(train[columns],train['registered'])
predictions2=model.predict(test[columns])
predictions=predictions1+predictions2
mse=mean_squared_error(test['cnt'],predictions)
print("RMSE using ExtraTreesRegressor:    ",end='')
rmse=math.sqrt(mse)
print(rmse,'\n')
#columns=['ind','cnt']
#sub=pd.DataFrame({'ind':rodes['ind'],'cnt':predictions})
#sub.to_csv('I:/day/predictions.csv',index=False,columns=columns)


model=GradientBoostingRegressor(n_estimators=99)
#model.fit(train[columns],train['cnt'])
#predictions=model.predict(test[columns])
#model.fit(train[columns],train['cnt'])
#predictions=model.predict(test[columns])

model.fit(train[columns],train['casual'])
predictions1=model.predict(test[columns])
predictions1=predictions1+0.5
predictions1=numpy.round(predictions1)
columns1=['ind','casual']
sub=pd.DataFrame({'ind':rodes['ind'],'casual':predictions1})
sub.to_csv('I:/day/predictions1.csv',index=False,columns=columns1)
model.fit(train[columns],train['registered'])
predictions2=model.predict(test[columns])
predictions2=predictions2+0.5
predictions2=numpy.round(predictions2)
columns2=['ind','registered']
sub=pd.DataFrame({'ind':rodes['ind'],'registered':predictions2})
sub.to_csv('I:/day/predictions2.csv',index=False,columns=columns2)
predictions=predictions1+predictions2
predictions=predictions+0.5
predictions=numpy.round(predictions)
mse=mean_squared_error(test['cnt'],predictions)
print("RMSE using GradientBoostingRegressor:    ",end='')
rmse=math.sqrt(mse)
print(rmse,'\n')
columns3=['ind','cnt']
sub=pd.DataFrame({'ind':rodes['ind'],'cnt':predictions})
sub.to_csv('I:/day/predictions.csv',index=False,columns=columns3)

#par=range(80,100,10)
#results=[]
#for i in percentiles:
#    fs=feature_selection.SelectPercentile(feature_selection.mutual_info_regression,percentile=i)
#    X_train=fs.fit_transform(train[columns],train['cnt'])
#    model.fit(X_train,train['cnt'])
#    X_test=fs.transform(test[columns])
#    predictions=model.predict(X_test)
#    mse=mean_squared_error(test['cnt'],predictions)
#    results=np.append(results,mse)
#print(results)
#opt=np.where(results==results.max())[0]
#print('Optimal number of feature %d '%percentiles[opt])
#columns=['ind','cnt']
#quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
## Store scalings in a dictionary so we can convert back later
#scaled_features = {}
#for each in quant_features:
#    mean, std = data[each].mean(), data[each].std()
#    scaled_features[each] = [mean, std]
#    data.loc[:, each] = (data[each] - mean)/std
#    # Save data for approximately the last 21 days 
#test_data = data[-21*24:]
## Now remove the test data from the data set 
#data = data[:-21*24]
#
## Separate the data into features and targets
#target_fields = ['cnt', 'casual', 'registered']
#features, targets = data.drop(target_fields, axis=1), data[target_fields]
#test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]
#train_features, train_targets = features[:-60*24], targets[:-60*24]
#val_features, val_targets = features[-60*24:], targets[-60*24:]