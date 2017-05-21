# -*- coding: utf-8 -*-
"""
Created on Wed May 10 23:06:59 2017

@author: SHRADDHA
"""

#import Section
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Part 1 DATA PREPROCESSING
dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

#Encoding the Categorical data
#Using the sklearn library, it automatically gives it indexes just by fitting
#OneHotEncoder makes different no of columns for differentiating / dumpy variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

X = X[:,1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state = 0)

#Feature Scaling
#Standardization
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)




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
classifier.add(Dense(units=6, kernel_initializer = "uniform", 
                     activation = "relu", input_dim=11))

#Adding the second hidden layer same for all, only input not required
classifier.add(Dense(units=6, kernel_initializer = "uniform", 
                     activation = "relu"))

#Adding the final layer or Output layer
classifier.add(Dense(units=1, kernel_initializer = "uniform", 
                     activation = "sigmoid"))

#Time to compile the model
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                   metrics = ['accuracy'])

#Most Interesting Step here
#Fitting the data
classifier.fit(X_train,y_train, batch_size=10 ,epochs = 100)




#Part 3 PREDICTING THE TEST RESULTS
#Predictions
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

X_c = np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])
ans = classifier.predict(sc_X.transform(X_c))
ans = (ans > 0.5)

#Make Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)



#--------------------------------------------------------------------------
#This portion can be done indivually with preprocessing
#ALL THIS TAKES TIME
#High Computational tasks
#Requires more configuration
#PART 4 EVALUATING, IMPROVING AND TUNING THE MODEL FOR BETTER PERFORMANCE
#Evaluating the ANN
import keras
'''from keras.models import Sequential #To Select model and train
from keras.layers import Dense #For Difference layers'''
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.layers import Dropout
def build_classifier():
    classifier = Sequential()
    #Adding hidden layer with dropout
    classifier.add(Dense(units=6, kernel_initializer = "uniform", 
                     activation = "relu", input_dim=11))    
    classifier.add(Dropout(p=0.1))
    classifier.add(Dense(units=6, kernel_initializer = "uniform", 
                     activation = "relu"))
    classifier.add(Dense(units=1, kernel_initializer = "uniform", 
                     activation = "sigmoid"))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                   metrics = ['accuracy'])
    return classifier

#Make a global classifier
classifier = KerasClassifier(build_fn = build_classifier,
                             batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train,
                             y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()



#Improving the ANN
#By adding Dropouts

#Parameter Tuning MOST FUN
#Will take a lot a lot of time !
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.layers import Dropout
def build_classifier():
    classifier = Sequential()
    #Adding hidden layer with dropout
    classifier.add(Dense(units=6, kernel_initializer = "uniform", 
                     activation = "relu", input_dim=11))    
    classifier.add(Dropout(p=0.1))
    classifier.add(Dense(units=6, kernel_initializer = "uniform", 
                     activation = "relu"))
    classifier.add(Dense(units=1, kernel_initializer = "uniform", 
                     activation = "sigmoid"))
    classifier.compile(optimizer = 'adam' #optimizer
                       , loss = 'binary_crossentropy',
                   metrics = ['accuracy'])
    return classifier

#Make a global classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = { 'batch_size': [25,32], 'epochs':[100,200], 
              'optimizer': {'adam','rmsprop'}}

grid_search = GridSearchCV(estimators = classifier, scoring = 'accracy',
                           param_grid = parameters, cv=10)
grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = gridsearch.best_score_


