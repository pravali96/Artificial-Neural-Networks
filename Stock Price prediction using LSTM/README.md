# Apple Stock Price Prediction Using Stacked LSTM Networks
The goal is to predict Apple's closing stock price for next 30 days based on past 100 days' close prices. In order to perform the analysis, extracted time-series data from 2016-06-27 to 2021-06-25 from TIINGO API.
### Data Source
https://www.tiingo.com/
### Data Processing
Extracted the closing prices from the data and reset the indexes, applied MinMaxScaler to bring the prices between 0 and 1. 
### Creating train_test_split
Since its a time series data, I have the split the data into 2 halfs- train data consisted of all the records indexed from 1 to 818 and the test data had indexes from 819 to 1259 (441 records).
The next step is to create X and y for the analysis. I have taken time_steps of 100. which mean for the 1st record: take 0 to 99 indexes as X and 100th index as output y and then repeat until we reach the last index. I repeated the same steps on train and test data. 
At the end of the process. X_train data has (717, 100) and y_train has (717,) and  X_test has (340, 100).
### Model Building
Reshaped X_train and X_test to 3 Dimensions before passing it to LSTM.
Added layers of LSTM and compiled the model using adam optimizer and calculated loss using MSE.
### Performance Metrics
Made predictions on both train and test data. Performed inverse scaling and calculated RMSE for both train and tests. RMSE for train set is 173.2 and RMSE for test set is 199.5
Plotted the values of Train predict and test predict.
### Predict the closing price for next 30 days
To continue predicting for 442nd record to 471st record, I followed the same process as above by taking timesteps of 100. I took 100 records, predict y_hat and then shift to right, add that y_hat to input and predict again for the following day and so on.
Passed all the records through a for loop to split the data and predict the values and added y_hats to a list.
Finally plotted the results for the next 30days.


