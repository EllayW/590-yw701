# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 14:24:41 2021

@author: Ella
"""
import random
random.seed(1001)
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.pipeline import make_pipeline as mp

class Read_Json:
        
    def __init__(self, path):
        f = open(path)

        data = json.load(f)
        
        f.close()
        
        self.df = pd.DataFrame.from_dict(data).iloc[:,2:]
        
        self.df = np.array(self.df)
        #self.age = self.df['age']
        #self.weight = self.df['weight']
        #self.is_adult = self.df['is_adult']
        
if __name__ == '__main__':
    data_weight = Read_Json('/home/ella/590-CODES/DATA/weight.json').df
    data_under_18 = np.where(data_weight[:,1] < 18)[0]
    
    ## visualize the data point by age and weight
   # Read_Json('weight.json').dot_plot(data_weight)
    #plt.ylabel('weight')
    #plt.xlabel('age')
    #plt.title('Dot plot with Age and Weight')
    #plt.show()

    # train-test splitting:
    #X,y = Read_Json('weight.json').normalizing(data_weight)
    X = data_weight[:,1:]
    y = data_weight[:,0]
    
    sc = StandardScaler()
    sc_fit1 = sc.fit(X[:,0].reshape(-1,1))
    sc_fit2 = sc.fit(X[:,1].reshape(-1,1))
    sc_trans1 = sc_fit1.transform(X[:,0].reshape(-1,1))
    sc_trans2 = sc_fit2.transform(X[:,1].reshape(-1,1))

    

    X_train_18, X_test_18, y_train_18, y_test_18 = train_test_split(sc_trans1[data_under_18].reshape(-1,1),
                                                                    sc_trans2[data_under_18].reshape(-1,1),
                                                                    test_size = 0.2, 
                                                                    random_state = 0)
    
    ####################################################################
    ############# Linear regression between Age and Weight:#############
    ####################################################################
    
    #### loss function -- mse:
    olr_mse = lambda m,b,x,y: np.sum((y-m*x-b)**2)/(y.shape[0])
    olr_mae = lambda m,b,x,y: np.sum(np.abs(y-m*x-b))/(y.shape[0])
    #### optimization of m,b:
    olr = minimize(lambda coef: olr_mse(*coef, X_train_18,y_train_18), 
                   x0=[1,1])
    
    while olr.success == False:
        thred = np.random.rand(2)
        olr = minimize(lambda coef: olr_mse(*coef, X_train_18,y_train_18), 
                   x0=thred)
    m,b = olr.x
    olr_pred = m*X_test_18+b
    
    
    ## visualize the boundary of train dataset:
    inversed_x = sc_fit1.inverse_transform(X_train_18)
    inversed_y = sc_fit2.inverse_transform(y_train_18)
    inversed_pred = sc_fit2.inverse_transform(m*X_train_18+b)
    
    ## measure of success
    mse_olr = olr_mse(m,b,sc_fit1.inverse_transform(X_test_18),
                      sc_fit2.inverse_transform(y_test_18))
    mae_olr = olr_mae(m,b,sc_fit1.inverse_transform(X_test_18),
                      sc_fit2.inverse_transform(y_test_18))
            
    fig, ax = plt.subplots()
    ax.scatter(data_weight[:,1],data_weight[:,2], marker = 'o')
    ax.plot(inversed_x,inversed_pred,color = 'black')
    plt.ylabel('weight')
    plt.xlabel('age')
    plt.text(70,80,'test MSE:{}'.format(np.round(mse_olr,3)))
    plt.text(70,70,'test MAE:{}'.format(np.round(mae_olr,3)))
    plt.legend(['prediction line','total dataset'])
    plt.title('Linear Regression with Age and Weight')
    plt.show()
    
    
    ##################################################################
    ############## Logistic regression between Age and Weight:########
    ##################################################################
    
    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(sc_trans1.reshape(-1,1),
                                                                    sc_trans2.reshape(-1,1),
                                                                    test_size = 0.2, 
                                                                    random_state = 0)
    
    #### loss function mse, mae:
    
    f1 = lambda x,w,x0: 1 + np.exp(-(x-x0)/w)
    logreg_f = lambda x,A,w,x0,s: A/(1+f1(x,w,x0))+s
    logreg_mse = lambda x,y,A,w,x0,s: np.sum((y-logreg_f(x,A,w,x0,s))**2)/(y.shape[0])
    logreg_mae = lambda x,y,A,w,x0,s: np.sum(np.abs(y-logreg_f(x,A,w,x0,s)))/(y.shape[0])
    
    logreg_loss_mse = lambda params: logreg_mse(X_train_2,y_train_2,
                                                                 *params)
    logreg = minimize(logreg_loss_mse, [.5,0.5,.5,0.5])
    
    
    while logreg.success == False:
        thred = np.random.rand(4)
        logreg = minimize(logreg_loss_mse, thred)
        
    A,w,x0,s = logreg.x
    logreg_pred = logreg_f(X_test_2, A,w,x0,s)
    
    ## visualize the boundary of train dataset:
    inversed_x = sc_fit1.inverse_transform(X_train_2)
    inversed_y = sc_fit2.inverse_transform(y_train_2)
    inversed_pred = sc_fit2.inverse_transform(logreg_f(X_train_2,A,w,x0,s))
    
    ## measure of success
    mse_logreg = logreg_mse(sc_fit1.inverse_transform(X_test_2),
                            sc_fit2.inverse_transform(y_test_2),A,w,x0,s)
    mae_logreg = logreg_mae(sc_fit1.inverse_transform(X_test_2),
                            sc_fit2.inverse_transform(y_test_2),A,w,x0,s)
    
    orders = np.argsort(inversed_x.ravel())
    
    fig, ax = plt.subplots()
    ax.plot(inversed_x[orders],inversed_pred[orders],color = 'black',)
    ax.scatter(inversed_x,inversed_y, marker = 'o')#, c = is_adult)
    plt.ylabel('weight')
    plt.xlabel('age')
    plt.text(70,80,'test MSE:{}'.format(np.round(mse_logreg,3)))
    plt.text(70,70,'test MAE:{}'.format(np.round(mae_logreg,3)))
    plt.legend(['prediction line','train dataset'])
    plt.title('Logistic Regression with Age and Weight')
    plt.show()
    
    ##################################################################
    ######## Logistic regression between Is_adult and Weight:#########
    ##################################################################
    
    ## normalizing
    X = data_weight[:,2]
    y = data_weight[:,0]
    
    sc = StandardScaler()
    sc_fit1 = sc.fit(X.reshape(-1,1))
    sc_trans1 = sc_fit1.transform(X.reshape(-1,1))

    X_train, X_test, y_train, y_test = train_test_split(sc_trans1.reshape(-1,1),
                                                                    y.reshape(-1,1),
                                                                    test_size = 0.2, 
                                                                    random_state = 0)
    
    #### loss function mse, mae, similar to the second model:
    classifier_logreg_loss_mse = lambda params: logreg_mse(X_train,y_train,
                                                                 *params)
    
    classifier_logreg = minimize(classifier_logreg_loss_mse, [.5,0.5,.5,0.5])
    
    
    while classifier_logreg.success == False:
        thred = np.random.rand(4)
        classifier_logreg = minimize(classifier_logreg_loss_mse, thred)
        
    A,w,x0,s = classifier_logreg.x
    classifier_logreg_pred = logreg_f(X_test, A,w,x0,s)
    
    ## visualize the boundary of train dataset:
    inversed_x = sc_fit1.inverse_transform(X_train)
    inversed_y = y_train
    inversed_pred = logreg_f(X_train,A,w,x0,s)
    
    ### measure of success
    mse_classifier_logreg = logreg_mse(sc_fit1.inverse_transform(X_test),
                                       y_test,A,w,x0,s)
    mae_classifier_logreg = logreg_mae(sc_fit1.inverse_transform(X_test),
                                       y_test,A,w,x0,s)
    
    orders = np.argsort(inversed_x.ravel())
            
    fig, ax = plt.subplots()
    ax.plot(inversed_x[orders], inversed_pred[orders], color='black')
    ax.scatter(inversed_x,inversed_y, marker = 'o')#
    plt.ylabel('Adult(1) Child(0)')
    plt.xlabel('weight')
    plt.text(25,.8,'test MSE:{}'.format(np.round(mse_classifier_logreg,3)))
    plt.text(25,.7,'test MAE:{}'.format(np.round(mae_classifier_logreg,3)))
    plt.legend(['prediction line','train dataset'])
    plt.title('Logistic Regression with Is_adult and Weight')
    plt.show()
    
    
