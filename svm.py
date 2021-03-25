# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 14:00:18 2021

@author: Tharun Loknath
"""
import numpy as np

class svm:
    def __init__(self,learning_rate=0.1,lamb=0.4,epoch=10000):
        self.lr=learning_rate
        self.lambd=lamb
        self.epooch=epoch
        self.b=None
        self.w=None
        
    def fit(self,X,Y):
        # 
        rows ,col =X.shape
        self.w=np.zeros(col)
        self.b=0
        y_dash=np.where(Y<=0,-1,1)
        
        for i in range(self.epooch):
            for idx,x in enumerate(X):
                yfx=(y_dash[idx]*np.dot(x,self.w)-self.b)
                if yfx>=1:
                    self.w-=self.lr*(2*self.lambd*self.w)
                else:
                    self.w-=self.lr(2*self.lambd*self.w-(yfx))
                    self.b-=self.lr*y_dash[idx]
                    
    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)
        
