# -*- coding: utf-8 -*-
"""
Created on Wed May 10 23:06:59 2017

@author: SHRADDHA
"""

#import Section
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                    random_state = 0)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

#Predictions
y_pred = classifier.predict(X_test)


#Make Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

