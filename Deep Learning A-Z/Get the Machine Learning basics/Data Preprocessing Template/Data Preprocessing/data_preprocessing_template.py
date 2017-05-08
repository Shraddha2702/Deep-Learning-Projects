#Data Preprocessing Template

#import Section
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Creating a dataset from pandas - ie dataframes
#For importing we use pandas library
dataset = pd.read_csv('Data.csv')

#Now we need to seperate our IV and DV
#X being independent variables array and y being array of DV

#Import as dataframes with pandas and changing into numpy array !

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Handling the missing data

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])


#Encoding the Categorical data
#Using the sklearn library, it automatically gives it indexes just by fitting

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:, 0])

#Stop from comparing the machine to think
#Do Dumpy Encoding, make three different columns
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()


#Repeating for y, not need to use onehotencoder since now it has only two values
#machine recognizes it
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)



#Splitting into Training and testing datasets

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state = 0)


#Feature Scaling
#Here FS is not required for Y, But Regressions need it
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

