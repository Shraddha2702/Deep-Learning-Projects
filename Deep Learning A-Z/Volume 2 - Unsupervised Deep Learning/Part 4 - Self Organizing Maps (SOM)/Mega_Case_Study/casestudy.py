#MEGA CASE STUDY
#Combine Supervised and Unsupervised Learning
#Going from Unsupervised to Supervised Learning
#By USL, we learn the patterns and then predict using SL
#2 Parts in these module

#Mega Case Study - Make a hybrid Deep Learning Module

#Part 1 - Identify the frauds with the self-organizing Maps
#Self Organizing Maps

#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import database
dataset= pd.read_csv('Credit_Card_Applications.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(X)

#Training the SOM
#minisom.py is the implementation of SOM in python which is opensource
from minisom import MiniSom 
som = MiniSom(x = 10, y = 10, input_len= 15, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

#Visualizing the Results
from pylab import bone, pcolor, colorbar, plot, show
#To initialize the window
bone()
pcolor(som.distance_map().T)
colorbar()

markers= ['o','s']
colors = ['r','g']

for i,x in enumerate(X):
    w = som.winner(x)
    plot(w[0]+0.5,w[1]+0.5,markers[y[i]], markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize= 10, markeredgewidth=2)
show()

mappings = som.win_map(X)

#To get the potential cheater
#Find the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(2,5)],mappings[(3,6)]),axis=0)
frauds = sc.inverse_transform(frauds)

#------------------------------------------------------------------------------
#Part 2 - Going from UnSupervised to Supervised Learning
#Creating the matrix of features
customers = dataset.iloc[:,1:].values

#Creating the dependent variable
is_fraud = np.zeros(len(dataset))

for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] = 1


#ANN Build
#Feature Scaling
#Standardization
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
customers = sc_X.fit_transform(customers)


#Part 2 NOW LETS MAKE ANN !!!

#First just import keras libraries and packages
import keras
from keras.models import Sequential #To Select model and train
from keras.layers import Dense #For Difference layers

#Initializing the ANN by defining the layers
#DOne using Squential model
#Make an object of Sequential class, which will be our ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(units=2, kernel_initializer = "uniform", 
                     activation = "relu", input_dim=15))


#Adding the final layer or Output layer
classifier.add(Dense(units=1, kernel_initializer = "uniform", 
                     activation = "sigmoid"))

#Time to compile the model
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                   metrics = ['accuracy'])

#Most Interesting Step here
#Fitting the data
classifier.fit(customers,is_fraud, batch_size=1 ,epochs = 2)




#Part 3 PREDICTING THE PROBABILITES OF FRAUDS
#Predictions
y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:,0:1].values, y_pred),axis=1)
#Sorting the y_pred
y_pred = y_pred[y_pred[:,1].argsort()]

