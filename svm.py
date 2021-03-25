# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 14:00:18 2021

@author: Tharun Loknath
"""
import numpy as np
import matplotlib.pyplot as plt

class svm:
    def __init__(self,learning_rate=0.001,lamb=0.1,epoch=10000):
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
                yfx=(y_dash[idx]*np.dot(x,self.w.T))+self.b-1 # (yi*(xi*w)-b-1)
                #print(yfx)
                if np.any(yfx>=1):
                    self.w-=self.lr*(2*self.lambd*self.w)
                    # w-=w-learningrate*(2*lambda*weight)
                else:
                    self.w-=self.lr*(2*self.lambd*self.w-(yfx))
                    # w-=w-learningrate*(2*lambda*weight-(yi*(xi*w)-b))
                    self.b-=self.lr*y_dash[idx]
                    # b-=learningrate*yi
        print(np.linalg.norm(self.w))
              
              
    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)
        
    
    
sv=svm()
x=np.array([[3,1],[3,-1],[7,1],[8,0],[1,0],[0,1],[-1,0],[-2,0]])
y=np.array(([0,0,0,0,1,1,1,1]))
sv.fit(x,y)


plt.plot(x, linestyle = 'dotted')
plt.show()
print(sv.predict([2.5,0.5]))
