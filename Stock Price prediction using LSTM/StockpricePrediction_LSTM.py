# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 12:36:12 2021

@author: prava
"""
import pandas as pd
import pandas_datareader as pdr
import numpy as np

key='bb947fb0385a03645ebe4c350063bee1a692e658'

# getting Apple Stock price Data
df= pdr.get_data_tiingo('AAPL', api_key=key)

df.to_csv('AAPL.csv')

df=pd.read_csv('AAPL.csv')

df.head()

df.columns

print(min(df.date), max(df.date))

df.shape #1258x14

# Performing the analysis to predict closing price 
df1= df.reset_index()['close']

df1.shape

import matplotlib.pyplot as plt
plt.plot(df1)

# LSTM are sensitive to the state of the data. so we apply
# MinMax Scaler
import numpy as np
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
df1= scaler.fit_transform(np.array(df1).reshape(-1,1))
df1

# Train-Test_split
# Incase of time series data, our data split must not be random, it should be ordered by date
training_size=int(len(df)*0.65)
test_size= len(df1)-training_size
train_data, test_data = df1[0:training_size,:], df1[training_size:len(df1), :1]

# Preprocessing Data
len(train_data)
len(test_data)

# Time steps- how many previous records are needed to make a reliable prediction
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)

# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

print(X_train.shape[1])
print(y_train.shape)

print(X_test.shape)
print(ytest.shape)

# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
print(X_train.shape)

# Create the Stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

model=Sequential()
model.add(LSTM(50,return_sequences=True, input_shape=(100,1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

model.summary()

model.fit(X_train, y_train, validation_data=(X_test, ytest), epochs=100, batch_size=64, verbose=1)

# predict values
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

##Transforming to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

# Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))
# Test Data RMSE
math.sqrt(mean_squared_error(ytest,test_predict))

# shift train predictions for plotting
look_back=100
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

# Predicting the output for next 30 days
len(test_data) #441
#In order to predict 442nd record, we need previos 100 records i.e., 341 to 441

x_input=test_data[341:].reshape(1,-1)
x_input.shape

#Then we convert values to a list and extract values
temp_input=list(x_input)
temp_input=temp_input[0].tolist()
len(temp_input) # Previous 100 days records

#Then we make predictions by taking 100 records at once
# we take 100 records, predict y_hat and then shift to right, add that y_hat to input and predict again for the following day and so on
# we predict for the next 30 days
from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
print(lst_output)

# Plotting
day_new=np.arange(1,101)
day_pred=np.arange(101,131)
plt.plot(day_new,scaler.inverse_transform(df1[1159:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))

#Extend the graphs
df3=df1.tolist()
df3.extend(lst_output)
plt.plot(df3[1200:])

#Perfrom inverse tansform on scaled values
df3=scaler.inverse_transform(df3).tolist()
plt.plot(df3)
