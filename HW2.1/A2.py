# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 07:26:10 2021

@author: Ella
"""


import random
random.seed(1002)
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#from scipy.optimize import minimize
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
    
    #### functions:
    
    f1 = lambda x,w,x0,s: np.exp(-(x-x0)/w)
    logreg_f = lambda x,A,w,x0,s: A/(1+f1(x,w,x0,s))+s
    logreg_sse = lambda x,y,A,w,x0,s: np.sum((y-logreg_f(x,A,w,x0,s))**2)
    
    ## optimizing function:
    iterations=[]
    loss_train=[]
    loss_val=[]
    iteration=0
    
    def training_sets(objective,method):
        X = objective[0] ## batch
        y = objective[1]
        if method == 'mini-batch':
            choose = random.sample(list(range(objective[0].shape[0])), 
                                   int(objective[0].shape[0]/2))
            X_us = X[choose,:]
            y_us = y[choose,:]
        elif method == 'stochastic':
            choose = random.sample(list(range(objective[0].shape[0])), 
                                   1)
            X_us = X[choose,:]
            y_us = y[choose,:]
        elif method == 'batch':
            X_us = X
            y_us= y
        
        return([X_us,y_us])
    
    
    def GD(objective,method,LR=0.001):

        global iteration,iterations,loss_train,loss_val
        
        
        #parameters:
        
        dx=0.001													
        t=0 	 							
        tmax=100000				
        tol = 10**-5
        NDIM = 4
        #xi =np.random.uniform(np.min(X_use),np.max(X_use),NDIM) ## initial guess
        xi = np.array([ 3,0.08,-3.8,-3.44])
        #xi = np.random.randn(NDIM)
        
        ## loss function:
        f = lambda X_use2,y_use2,params:logreg_sse(X_use2,
                                                 y_use2,
                                                 *params)/len(X_use2)

        while(t<=tmax):
            t=t+1
            X_use1,y_use1 = training_sets(objective,method)
            
            #if X_use1.shape[0] != X_train_2.shape[0]:
                #X_use1,y_use1 = training_sets(objective,method,it=t)
                
                
        	# gradient (shape len(x_use) x 1)
            df_dx=np.zeros(NDIM)
            
            for i in range(0,NDIM):
                dX=np.zeros(NDIM)
                dX[i]=dx
                xm1=xi-dX
                df_dx[i]=(f(X_use1,y_use1,xi)-f(X_use1,y_use1,xm1))/dx
                
                xip1=xi-LR*df_dx 
                

            if(t%100==0):
                df=np.mean(np.absolute(f(X_use1,y_use1,xip1)-f(X_use1,y_use1,xi)))
                #print('diff:',df)
                print(t,"iteration",'loss:',f(X_use1,y_use1,xi), 
                      "diff:",df)
                
                #print('xi',xip1)
        
                if(df<tol):
                    print("STOPPING CRITERION MET (STOPPING TRAINING)")
                    print(df)
                    break
        
            else:
                
                xi=xip1
                
                training_loss = logreg_sse(X_train_2,y_train_2,*xip1)
                val_loss = logreg_sse(X_validation_2,y_validation_2,*xip1)
                
                loss_train.append(training_loss)
                loss_val.append(val_loss)
                iterations.append(iteration)
                iteration+=1
        return(xi)
    
    def MGD(objective,method,LR=0.001,alpha = 0.001):

        global iteration,iterations,loss_train,loss_val,delta
        delta = []
        #X_use,y_use = training_sets(objective,method,it=0)
        #parameters:
        
        dx=0.001													
        t=0 	 							
        tmax=100000				
        tol = 10**-5
        NDIM = 4
        delta.append(np.zeros(NDIM))
        #xi =np.random.uniform(np.min(X_use),np.max(X_use),NDIM) ## initial guess
        xi = np.array([ 3,0.08,-3.8,-3.44])
        #xi = np.random.randn(NDIM)
        
        ## loss function:
        f = lambda X_use,y_use,params:logreg_sse(X_use,
                                                 y_use,
                                                 *params)/len(X_use)

        while(t<=tmax):
            t=t+1
            X_use,y_use = training_sets(objective,method)
            #if X_use.shape[0] != X.shape[0]:
             #   X_use,y_use = training_sets(objective,method,it=t)
                
                
        	# gradient (shape len(x_use) x 1)
            df_dx=np.zeros(NDIM)
            
            
            for i in range(0,NDIM):
                dX=np.zeros(NDIM)
                dX[i]=dx
                xm1=xi-dX
                df_dx[i]=(f(X_use,y_use,xi)-f(X_use,y_use,xm1))/dx
                
                xip1=xi-LR*df_dx-alpha *delta[-1]
            delta.append(df_dx)
            
            if(t%100==0):
                df=np.mean(np.absolute(f(X_use,y_use,xip1)-f(X_use,y_use,xi)))
                #print('diff:',df)
                print(t,"iteration",'loss:',f(X_use,y_use,xi), 
                      "diff:",df)
                
                #print('xi',xip1)
        
                if(df<tol):
                    print("STOPPING CRITERION MET (STOPPING TRAINING)")
                    print(df)
                    break
        
            else:
                
                xi=xip1
                
                training_loss = logreg_sse(X_train_2,y_train_2,*xip1)
                val_loss = logreg_sse(X_validation_2,y_validation_2,*xip1)
                
                loss_train.append(training_loss)
                loss_val.append(val_loss)
                iterations.append(iteration)
                iteration+=1
        return(xi)
    
    
    def optimizer(objective, method,algo = 'GD',LR=0.001):
        global met, alg
        met = method
        alg = algo
        if algo == 'GD':
            func = GD(objective = objective, method = method, LR= LR)
        elif algo =='MGD':
            func = MGD(objective = objective, method = method, LR = LR)
            
        return(func)
            
    ######################### plug in data: ####################
  
    data_weight = Read_Json('weight.json').df
    
    # train-test splitting:
    #X,y = Read_Json('weight.json').normalizing(data_weight)
    X = data_weight[:,1:]
    y = data_weight[:,0]
    
    sc = StandardScaler()
    sc_fit1 = sc.fit(X[:,0].reshape(-1,1))
    sc_fit2 = sc.fit(X[:,1].reshape(-1,1))
    sc_trans1 = sc_fit1.transform(X[:,0].reshape(-1,1))
    sc_trans2 = sc_fit2.transform(X[:,1].reshape(-1,1))

    ##################################################################
    ############## Logistic regression between Age and Weight:########
    ##################################################################
    
    ## split the df into train,validation & test
    
    X_train, X_test_2, y_train, y_test_2 = train_test_split(
        sc_trans1.reshape(-1,1),
        sc_trans2.reshape(-1,1),
        test_size = 0.2,
        random_state = 0)
    
    X_train_2, X_validation_2, y_train_2, y_validation_2 = train_test_split(
        X_train.reshape(-1,1),
        y_train.reshape(-1,1),
        test_size = 0.1,
        random_state = 0)
    
   #################### visualize the train/test loss################
    
    iterations=[]
    loss_train=[]
    loss_val = []
    iteration=0
    
    A,w,x0,s = optimizer([X_train_2,y_train_2],algo = 'GD',method = 'batch')
    
    ## training & validation errors
    plt.scatter(iterations,loss_train,c = 'r',marker = 'o')
    plt.scatter(iterations,loss_val, c = 'g',marker = 'o')
    plt.xlabel('optimizer iterations')
    plt.ylabel('loss')
    plt.title('Train/Validation Error in Logistic Regression with Age and Weight by '+ alg+ ' and '+ met)
    plt.legend(['Train Loss', 'Validation Loss'])
    plt.show()
    
    logreg_pred = logreg_f(X_train_2, A,w,x0,s)
    logreg_pred_val = logreg_f(X_validation_2, A,w,x0,s)
    
    
    ############## visualize the boundary of train dataset:#############
    
    inversed_x_train = sc_fit1.inverse_transform(X_train_2)
    inversed_y_train = sc_fit2.inverse_transform(y_train_2)
    inversed_x_val = sc_fit1.inverse_transform(X_validation_2)
    inversed_y_val = sc_fit2.inverse_transform(y_validation_2)
    inversed_x_test = sc_fit1.inverse_transform(X_test_2)
    inversed_y_test = sc_fit2.inverse_transform(y_test_2)
    inversed_pred = sc_fit2.inverse_transform(logreg_f(X_train_2,A,w,x0,s))
    
    orders = np.argsort(inversed_x_train.ravel())
    
    fig, ax = plt.subplots()
    ax.plot(inversed_x_train[orders],inversed_pred[orders],color = 'black',)
    ax.scatter(inversed_x_train,inversed_y_train, marker = 'o')
    ax.scatter(inversed_x_val,inversed_y_val, marker = 'x')
    ax.scatter(inversed_x_test,inversed_y_test, marker = '*')
    plt.ylabel('weight')
    plt.xlabel('age')
    plt.legend(['prediction line','train','validation','test'])
    plt.title('Logistic Regression with Age and Weight by '+ alg+ ' and '+ met)
    plt.show()
    
    ################## visualize the y_pred and y_data:##############
    
    fig, ax = plt.subplots()
    ax.scatter(logreg_pred,y_train_2, marker = 'o')
    ax.scatter(logreg_pred_val,y_validation_2, marker = 'x')
    plt.ylabel('y_pred')
    plt.xlabel('y_data')
    plt.legend(['train','validation'])
    plt.title('Prediction vs. True Data by method by '+ alg+ ' and '+ met)
    plt.show()
