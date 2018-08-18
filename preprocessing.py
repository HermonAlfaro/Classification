# class to do a pipeline preprocessing

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

# Data Preprocessing Template

class Preprocessing:
    def __init__(self, path):
        """
        input: path of the csv to read the data
        attributes: data: data in the csv
                    X: descriptive matrix 
                    y: array to describe
        """
        self.data = pd.read_csv(path)
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def creating_data(self,idx_X, idx_y):
        """
        input: idx1: List of index of data columns considered as the descriptive variables
                     It can also be a list of booleans    
               idx2: index of data column considered as the variable to descrive
        Output: The descriptive matrix X and the array to describe y 
        """
        self.X = self.data.iloc[:, idx_X].values
        self.y = self.data.iloc[:, idx_y].values
        
        return self.X, self.y

    def spliting_data(self,test_size, random_state,only_y):
        """
        Input: test_size: fraccion of the data considered to the test set
               random_state: level of randomness in the choose of the examples 
               to be considered in the test set
        Output: X_train, y_train, X_test, y_test
        """
        self.X_train, self.X_test, self.y_train, self.y_test =\
        train_test_split(self.X, self.y, test_size = test_size, random_state = random_state)
        if only_y:
            return self.y_train, self.y_test
        
        return self.X_train, self.X_test, self.y_train, self.y_test

    def feature_scaling(self,scaler):
        """
        Input: scaler: class to be used to to de scaling. 
               Written to use the scaler of scikit_learn but it is useful 
               for any class that works like the scikit-learn classes
        Output: X_train and X_test scaled by the X_train information
        """
        sc_X = scaler()
        self.X_train = sc_X.fit_transform(self.X_train)
        self.X_test = sc_X.transform(self.X_test)
        return self.X_train, self.X_test
    
    @staticmethod
    def preprocessing_pipeline(path,idx_X,idx_y,test_size,random_state,scaler):
        """
        Pipeline that uses the methods of the class to return the X_train, X_test scalered
        and y_train, y_test
        """
        
        X_train, X_test, y_train, y_test = [0,0,0,0]
        self = Preprocessing(path)
        self.creating_data(idx_X,idx_y)
        X_train, X_test,y_train, y_test = self.spliting_data(test_size,random_state,False)
        X_train, X_test = self.feature_scaling(scaler)
        
    
        return X_train, X_test, y_train, y_test
