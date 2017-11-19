# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 18:51:01 2017

@author: LZDSLI
"""
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor    
from sklearn.cross_validation import train_test_split
from sklearn import feature_selection
from sklearn.svm import SVR
import numpy 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
import pylab as pl
import seaborn as sns
data_path = 'I:/hour/train_hour.csv'

rides = pd.read_csv(data_path)
rides.head()
data_path = 'I:/hour/hour_out.csv'
rodes=pd.read_csv(data_path)
cnt_correlations=rides.corr()['cnt']
print("\n Reading success! cnt-correlations：\n")
print(cnt_correlations)
rides[:24*10].plot(x='dteday', y='cnt')
dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)
    dummies = pd.get_dummies(rodes[each], prefix=each, drop_first=False)
    rodes = pd.concat([rodes, dummies], axis=1)
cnt_correlations=rides.corr()['cnt']
print("\n Reading success! cnt-correlations：\n")
print(cnt_correlations)
#fields_to_drop = ['ind', 'dteday', 'season', 'weathersit', 
#                  'weekday',  'atemp','mnth',  'hr','holiday']
fields_to_drop = ['ind','dteday']
data = rides.drop(fields_to_drop, axis=1)
tesdata= rodes.drop(fields_to_drop, axis=1)
data.head()
tesdata.head()
#fig,axes = plt.subplots(nrows=3,ncols=2,figsize=(15,20))
#sns.boxplot(y="cnt",data=data,ax=axes[0][0])
#sns.boxplot(x="hr",y="cnt",data=data,ax=axes[0][1])
#sns.boxplot(x="weathersit",y="cnt",data=data,ax=axes[1][0])
#sns.boxplot(x="weekday",y="cnt",data=data,ax=axes[1][1])
#sns.boxplot(x="season",y="cnt",data=data,ax=axes[2][0])
#sns.boxplot(x="workingday",y="cnt",data=data,ax=axes[2][1])
#fig,axes = plt.subplots(nrows=4)


columns=list(tesdata.columns)
columns.remove('cnt')
columns.remove('casual')
columns.remove('registered')

#columns=['yr','temp','windspeed','season_1','season_3','weathersit_1','weathersit_3','hr_1','hr_2','hr_3','hr_4','hr_5','hr_6','hr_8','hr_16','hr_17','hr_18','hr_19','hr_23','workingday']
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
#print("MSE using LinearRegression:    ",end='')
#print(mse,'\n')
#model=SGDRegressor()
#model.fit(train[columns],train['casual'])
#predictions1=model.predict(test[columns])
#model.fit(train[columns],train['registered'])
#predictions2=model.predict(test[columns])
#predictions=predictions1+predictions2
#mse=mean_squared_error(test['cnt'],predictions)
#print("MSE using SGDRegression:    ",end='')
#print(mse,'\n')

#linear_svr=SVR(kernel='linear')
#linear_svr.fit(train[columns],train['cnt'])
#predictions=linear_svr.predict(test[columns])
#mse=mean_squared_error(test['cnt'],predictions)
#print("MSE using linear_svr:    ",end='')
#print(mse,'\n')
#poly_svr=SVR(kernel='poly')
#poly_svr.fit(train[columns],train['cnt'])
#predictions=poly_svr.predict(test[columns])
#print("MSE using poly_svr:    ",end='')
#print(mse,'\n')
#rbf_svr=SVR(kernel='rbf')
#rbf_svr.fit(train[columns],train['cnt'])
#predictions=rbf_svr.predict(test[columns])
#print("MSE using rbf_svr:    ",end='')
#print(mse,'\n')

#model=DecisionTreeRegressor(min_samples_leaf=5)
#model.fit(train[columns],train['casual'])
#predictions1=model.predict(test[columns])
#model.fit(train[columns],train['registered'])
#predictions2=model.predict(test[columns])
#predictions=predictions1+predictions2
#mse=mean_squared_error(test['cnt'],predictions)
#print("MSE using DecisionTreeRegression:    ",end='')
#print(mse,'\n')
#percentiles=range(1,100,2)
#results=[]
#for i in percentiles:
#    fs=feature_selection.SelectPercentile(feature_selection.chi2,percentile=i)
#    X_train=fs.fit_transform(train[columns],train['cnt'])
#    model.fit(X_train,train['cnt'])
#    X_test=fs.transform(test[columns])
#    predictions=model.predict(X_test)
#    mse=mean_squared_error(test['cnt'],predictions)
#    results=np.append(results,mse)
#print(results)
#opt=np.where(results==results.max())[0]
#print('Optimal number of feature %d' %percentiles[opt])


model=RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_split=1e-07, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=500, n_jobs=1, oob_score=False, random_state=None,
           verbose=0, warm_start=False)
model.fit(train[columns],train['casual'])
predictions1=model.predict(test[columns])
predictions1=predictions1+0.5
predictions1=numpy.round(predictions1)
columns1=['ind','casual']
sub=pd.DataFrame({'ind':rodes['ind'],'casual':predictions1})
sub.to_csv('I:/hour/predictions1.csv',index=False,columns=columns1)
model.fit(train[columns],train['registered'])
predictions2=model.predict(test[columns])
predictions2=predictions2+0.5
predictions2=numpy.round(predictions2)
columns2=['ind','registered']
sub=pd.DataFrame({'ind':rodes['ind'],'registered':predictions2})
sub.to_csv('I:/hour/predictions2.csv',index=False,columns=columns2)
predictions=predictions1+predictions2
predictions=predictions+0.5
predictions=numpy.round(predictions)
mse=mean_squared_error(test['cnt'],predictions)
print("MSE using RandomForsetRegression:    ",end='')
print(mse,'\n')
columns3=['ind','cnt']
sub=pd.DataFrame({'ind':rodes['ind'],'cnt':predictions})
sub.to_csv('I:/hour/predictions.csv',index=False,columns=columns3)


#model=ExtraTreesRegressor()
#model.fit(train[columns],train['casual'])
#predictions1=model.predict(test[columns])
#model.fit(train[columns],train['registered'])
#predictions2=model.predict(test[columns])
#predictions=predictions1+predictions2
#mse=mean_squared_error(test['cnt'],predictions)
#print("MSE using ExtraTreesRegressor:    ",end='')
#print(mse,'\n')
#percentiles=range(1,100,2)
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
#
#pl.plot(percentiles,results)
#pl.xlabel('percentile of feature')
#pl.ylabel('result')
#pl.show()
#fs=feature_selection.SelectPercentile(feature_selection.mutual_info_regression,percentile=percentiles[opt])
#X_train=fs.fit_transform(train[columns],train['cnt'])
#model.fit(X_train,train['cnt'])
#X_test=fs.transform(test[columns])
#predictions=model.predict(X_test)
#mse=mean_squared_error(test['cnt'],predictions)
#print(mse)
#sub=pd.DataFrame({'ind':rodes['ind'],'cnt':predictions})
#sub.to_csv('I:/hour/predictions2.csv',index=False,columns=columns)


#model=GradientBoostingRegressor()
#model.fit(train[columns],train['casual'])
#predictions1=model.predict(test[columns])
#model.fit(train[columns],train['registered'])
#predictions2=model.predict(test[columns])
#predictions=predictions1+predictions2
#mse=mean_squared_error(test['cnt'],predictions)
#print("MSE using GradientBoostingRegressor:    ",end='')
#print(mse,'\n')


# =============================================================================
# quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
# 
# scaled_features = {}
# for each in quant_features:
#     mean, std = data[each].mean(), data[each].std()
#     scaled_features[each] = [mean, std]
#     data.loc[:, each] = (data[each] - mean)/std
# =============================================================================
