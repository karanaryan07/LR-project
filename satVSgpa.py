# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 20:57:31 2019

@author: Lenovo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Simple linear regression.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 1/5, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

#graph of reality vs virtual of training set
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='green')
plt.title('SAT vs GPA(training set)')
plt.xlabel('SAT')
plt.ylabel('GPA')
plt.show()

#graph on test set
plt.scatter(X_test,y_test,color='blue')
plt.plot(X_test,y_pred,color='green')
plt.title('SAT vs GPA(test set)')
plt.xlabel('SAT')
plt.ylabel('GPA')
plt.show()


#graph of combined
plt.scatter(X_test,y_test,color='blue')
plt.plot(X_test,y_pred,color='green')
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='black')
plt.title('SAT vs GPA')
plt.xlabel('SAT')
plt.ylabel('GPA')
plt.show()


