import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix, accuracy_score

class Classifier:
    def __init__(self,clf,X_train,X_test,y_train,y_test):
        self.clf = clf
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred = None
        self.y_pred_floor = None
        self.y_pred_proba = None
        self.cm = None
        self.acc = None
        self.acc_floor = None
    
    def fitting(self,params):
        clf = self.clf
        clf = clf(**params)
        self.clf = clf.fit(self.X_train, self.y_train)
        return clf.fit(self.X_train, self.y_train)
    
    def predicting(self):
        
        self.y_pred = self.clf.predict(self.X_test)
        return self.y_pred
        
    def predict_proba(self):
        self.y_pred_proba = self.clf.predict_proba(self.X_test)
        return self.y_pred_proba
    
    def confusion_matrix(self):
        self.cm = confusion_matrix(self.y_test,self.y_pred)
        cm = self.cm
        print("         Confusion matrix : ")
        print(f"""
                           -------------------
                           |    predicted    |
                           |-----------------|
                           |1° class|2° class|
           -----------------------------------
           |    |  1° class|   {cm[0,0]}   |   {cm[0,1]}    |
           |Real|----------|--------|--------|
           |    |  2° class|   {cm[1,0]}    |   {cm[1,1]}   |  
           -----|--------- |--------|--------|
               """)
        return self.cm
    
    def accuracy(self):
        self.acc = accuracy_score(self.y_test,self.y_pred)
        print("accuracy :")
        print(accuracy_score(self.y_test,self.y_pred))
        return self.acc
            
    
    
    def colormap(self,idx1,idx2,is_train=True):
        """
        method to visualize the results of the classification in 2D. The color says that that belongs
        the data with the value of (X,y)
        Input: idx1: index of the column data that will be the x-axis of the graph
               idx2: index of the column data that will be the y-axis of the graph
               is_train: True: graphic with the train dat, False: with test set
               
        """
        classifier = self.clf
        if is_train:
            X_set, y_set = self.X_train, self.y_train
        else:
            X_set, y_set = self.X_test, self.y_test
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, idx1].min() - 1, stop = X_set[:, idx1].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, idx2].min() - 1, stop = X_set[:, idx2].max() + 1, step = 0.01))
        plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.75, cmap = ListedColormap(('red', 'green')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c = ListedColormap(('red', 'green'))(i), label = j)
        if is_train:
            title="train set"
        else:
            title="test set"
        plt.title(title)
        plt.xlabel("first variable descriptive")
        plt.ylabel("second variable descriptive")
        plt.legend()
        plt.show()
        
    
    
    @staticmethod
    def classification_pipeline(path,classifier,idx1,idx2,test_size,random_state,scaler,params):
        X_train, X_test, y_train, y_test = \
        Preprocessing.preprocessing_pipeline(path,idx1,idx2,test_size,random_state,scaler)
        self = Classifier(classifier,X_train,X_test,y_train,y_test)
        self.fitting(params)         
        self.predicting()
        self.confusion_matrix()
        self.accuracy()
        self.colormap(0,1,True)
        self.colormap(0,1,False)
        
        return self
        


