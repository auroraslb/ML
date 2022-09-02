# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 11:00:17 2021

@author: IST-AAut
"""

import numpy as np
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import balanced_accuracy_score as BACC

def scores(y_real,y_pred,mode):
    ###y_real - ground truth vector 
    ###y_pred - vector of predictions, must have the same shape as y_real
    ###mode   - if evaluating regression ('r') or classification ('c')
    
    if y_real.shape != y_pred.shape:
        print('confirm that both of your inputs have the same shape')
    else:
        if mode == 'r':
            mse = MSE(y_real,y_pred)
            print('The Mean Square Error is', mse)
            return mse
        
        elif mode == 'c':
            bacc = BACC(y_real,y_pred)
            print('The Balanced Accuracy is', bacc)
            return bacc
        
        else:
            print('You must define the mode input.')