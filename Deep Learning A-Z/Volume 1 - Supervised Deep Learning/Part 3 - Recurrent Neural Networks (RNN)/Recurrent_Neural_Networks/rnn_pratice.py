# -*- coding: utf-8 -*-
"""
08:08:21 2017

@author: SHRADDHA
"""

#Recurrent Neural Network

#PART 1 DATA PREPROCESSING

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing training set
training_set = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = training_set.iloc[:,1:2].values

#Feature Scaling
#Normalization used here where only min and max values are needed
#Once known min and max they are transformed
#Not Standardization
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

#Getting Input and Output
#In RNNs, here we take the input as the current price of Stocks today
#Using the current, we predict the next day's value
#That is the what happens, and hence we divide the ds into two parts
X_train = training_set[0:1257]
y_train = training_set[1:1258]

#Reshaping
#It's used by RNN   
X_train = np.reshape(X_train,(1257,1,1))


#PART 2 BUILDING THE RNNs

#Importing the libraries
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import LSTM

#Initializing the RNNs
regressor = Sequential()

#Adding the LSTM Layer and input layer
regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None,1)))
regressor.add(Dense(units = 1))

#Compile the Regressor, compile method used
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#Fitting the training set
regressor.fit(X_train, y_train, batch_size = 32, epochs = 200)


#PART 3 PREDICTING AND VISUALIZING THE TEST SET

#Importing the test_Set
test_set= pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = test_set.iloc[:,1:2].values

#Getting the predicted Stock Prices of 2017
inputs = real_stock_price
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (20,1,1))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#Visualizing the Stock Prices
plt.plot(real_stock_price, color = 'red', label = 'Real Stock Price')
plt.plot(predicted_stock_price , color = 'blue', label = 'Predicted Stock Price')
plt.title('Google Stock Price Predictions')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


#PART 4 EVALUATING THE RNN
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
