# -*- coding: utf-8 -*-
"""
Created on Thu May 11 22:19:07 2017

@author: SHRADDHA
"""

#import Section
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values