# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 23:15:13 2021

@author: Ella
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
import pandas as pd
import Seaborn_visualizer as SBV
from scipy.optimize import minimize

##############################################################
######################## Read the Data #######################
##############################################################

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

df = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)



#X=['Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration']
X_col=[1,2,3,4,5]
Y_col=[0]
XY_col=X_col+Y_col
X_keys =SBV.index_to_keys(df,X_col)        #dependent var
Y_keys =SBV.index_to_keys(df,Y_col)        #independent var
XY_keys=SBV.index_to_keys(df,XY_col)        #independent var


X=df[X_keys].to_numpy()
Y=df[Y_keys].to_numpy()

Xtmp=[]
Ytmp=[]

for i in range(0,X.shape[0]):
    if(not 'nan' in str(X[i])):
        Xtmp.append(X[i])
        Ytmp.append(Y[i])
X=np.array(Xtmp)
Y=np.array(Ytmp)

X_mean=np.mean(X,axis=0); X_std=np.std(X,axis=0)
Y_mean=np.mean(Y,axis=0); Y_std=np.std(Y,axis=0)
X2=(X-X_mean)/X_std

Y2=(Y-Y_mean)/Y_std


##############################################################
######################## Hyperparameter ######################
##############################################################

IPLOT=True
I_Normaliza = True
model_type = 'ANN'
model_type = 'linear'

PARADIGM = 'batch'
algo = 'GD'
LR = 0.1
iteration=1			#ITERATION COUNTER
dx=0.0001			#STEP SIZE FOR FINITE DIFFERENCE
max_iter=5000		#MAX NUMBER OF ITERATION
tol=10**-10			#EXIT AFTER CHANGE IN F IS LESS THAN THIS 
max_rand_wb=1

GAMMA_L1 = 0.0
GAMMA_L2 = 0.0001
alpha = 0.25

layers=[5,1] 
activations = ['identity']

## split the dataset
f_train=0.8; f_val=0.15; f_test=0.05;

if(f_train+f_val+f_test != 1.0):
	raise ValueError("f_train+f_val+f_test MUST EQUAL 1");

#PARTITION DATA
rand_indices = np.random.permutation(X.shape[0])
CUT1=int(f_train*X.shape[0]); 
CUT2=int((f_train+f_val)*X.shape[0]); 
train_idx, val_idx, test_idx = rand_indices[:CUT1], rand_indices[CUT1:CUT2], rand_indices[CUT2:]
print('------PARTITION INFO---------')
print("train_idx shape:",train_idx.shape)
print("val_idx shape:"  ,val_idx.shape)
print("test_idx shape:" ,test_idx.shape)

##############################################################
############################# Model ##########################
##############################################################
# WB: initial weights, bias

## split the weight into layers
### function required
def extract_submatrices(WB):
    submatrices=[]; K=0
    for i in range(0,len(layers)-1):
        Nrow=layers[i+1]; Ncol=layers[i] #+1
        w=np.array(WB[K:K+Nrow*Ncol].reshape(Ncol,Nrow).T) #unpack/ W 
        K=K+Nrow*Ncol; #print i,k0,K
        Nrow=layers[i+1]; Ncol=1; #+1
        b=np.transpose(np.array([WB[K:K+Nrow*Ncol]])) #unpack/ W 
        K=K+Nrow*Ncol; #print i,k0,K
        submatrices.append(w); submatrices.append(b)
        #print(w.shape,b.shape)
    return submatrices

## Activation Function:

def sigmoid(x):
    return(1/(1+np.exp(-x)))

def Tanh(x):
    return (1-np.exp(-2*x))/(1+np.exp(-2*x))

## functions required
def projection(x,p):
    ## p is intial weight
    ## result is y_prediction
    
    p_extract = extract_submatrices(p)
    #result = p[0]+np.matmul(x,p[1:].reshape(NFIT-1,1))
    result = x
    for i in range(len(layers)-1):
        weight = np.transpose(p_extract[2*i])
        bias = np.transpose(p_extract[2*i+1])
        bias = np.tile(bias,(len(x),1))
        #print(bias.shape)
        result = np.matmul(result,weight)+bias
        #print(result.shape)
        if activations[i]=="identity":
            result = result
            
        elif activations[i] == "sigmoid":
            result = sigmoid(result)
            
        elif activations[i] == 'Tanh':
            result = Tanh(result)

    return(result)
        
def model(x,p,model_type = 'linear'):
## model_type = ['linear','logistic','ANN']
    if model_type == 'linear':
        layers=[x.shape[1],1] 
        activations = ['identity']
        return(projection(x,p))
    elif model_type == 'logistic':
        ayers=[x.shape[1],1] 
        activations = ['sigmoid']
        return(projection(x,p))
    elif model_type == 'ANN':
        ayers=[x.shape[1],5,5,1] 
        activations = ['sigmoid','Tanh','identity']
        return(projection(x,p))

def predict(p):
	global YPRED_T,YPRED_V,YPRED_TEST,MSE_T,MSE_V
	YPRED_T=model(X2[train_idx],p)
	YPRED_V=model(X2[val_idx],p)
	YPRED_TEST=model(X2[test_idx],p)
	MSE_T=np.mean((YPRED_T-Y2[train_idx])**2.0)
	MSE_V=np.mean((YPRED_V-Y2[val_idx])**2.0)

##############################################################
####################### Loss Function ########################
##############################################################

## L1
def loss(p,index_2_use, gamma_L1 = 0.01, gamma_L2 = 0):
	errors=model(X2[index_2_use],p)-Y2[index_2_use]  #VECTOR OF ERRORS
	training_loss=np.mean(errors**2.0)+\
        gamma_L1*np.sum(p**2)+\
        gamma_L2*np.sum(np.abs(p))		#MSE
	return training_loss

##############################################################
########################### Minimizer ########################
##############################################################

def minimizer(f,xi, algo='GD', LR=0.01):
	global epoch,epochs, loss_train,loss_val 
	# x0=initial guess, (required to set NDIM)
	# algo=GD or MOM
	# LR=learning rate for gradient decent

	#PARAM
	iteration=1			#ITERATION COUNTER
	dx=0.0001			#STEP SIZE FOR FINITE DIFFERENCE
	max_iter=5000		#MAX NUMBER OF ITERATION
	tol=10**-10			#EXIT AFTER CHANGE IN F IS LESS THAN THIS 
	NDIM=len(xi)		#DIMENSION OF OPTIIZATION PROBLEM

	#OPTIMIZATION LOOP
	while(iteration<=max_iter):

		#DATASET PARITION BASED ON TRAINING PARADIGM
		#-------------------------
		if(PARADIGM=='batch'):
			if(iteration==1): index_2_use=train_idx
			if(iteration>1):  epoch+=1
		else:
			print("REQUESTED PARADIGM NOT CODED"); exit()

		df_dx=np.zeros(NDIM);	#INITIALIZE GRADIENT VECTOR
		for i in range(0,NDIM):	#LOOP OVER DIMENSIONS

			dX=np.zeros(NDIM);  #INITIALIZE STEP ARRAY
			dX[i]=dx; 			#TAKE SET ALONG ith DIMENSION
			xm1=xi-dX; 			#STEP BACK
			xp1=xi+dX; 			#STEP FORWARD 

			#CENTRAL FINITE DIFF
			grad_i=(f(xp1,index_2_use)-f(xm1,index_2_use))/dx/2

			# UPDATE GRADIENT VECTOR 
			df_dx[i]=grad_i 
			
		#TAKE A OPTIMIZER STEP
		if(algo=="GD"):  xip1=xi-LR*df_dx 
		if(algo=="MOM"): print("REQUESTED ALGORITHM NOT CODED"); exit()

		#REPORT AND SAVE DATA FOR PLOTTING
		if(iteration%100==0):
			predict(xi)	#MAKE PREDICTION FOR CURRENT PARAMETERIZATION
			print(iteration,"	",epoch,"	",MSE_T,"	",MSE_V) 

			#UPDATE
			epochs.append(epoch); 
			loss_train.append(MSE_T);  loss_val.append(MSE_V);

			#STOPPING CRITERION (df=change in objective function)
			df=np.absolute(f(xip1,index_2_use)-f(xi,index_2_use))
			if(df<tol):
				print("STOPPING CRITERION MET (STOPPING TRAINING)")
				break

		xi=xip1 #UPDATE FOR NEXT PASS
		iteration=iteration+1

	return xi


##############################################################
####################### Fit the Model ########################
##############################################################

#RANDOM INITIAL GUESS FOR FITTING PARAMETERS
#po=np.random.uniform(2,1.,size=NFIT)
max_rand_wb=1

## ANN parameter
## linear regression

NFIT=0; 
for i in range(1,len(layers)):
    print("Nodes in layer-",i-1," = ",layers[i-1])  
    NFIT=NFIT+layers[i-1]*layers[i]+layers[i]

po=np.random.uniform(-max_rand_wb,max_rand_wb,size=NFIT) 

PARADIGM='batch'

#SAVE HISTORY FOR PLOTTING AT THE END
epoch=1; epochs=[]; loss_train=[];  loss_val=[]

#TRAIN MODEL USING SCIPY MINIMIZ 
p_final=minimizer(loss,po)
print("OPTIMAL PARAM:",p_final)
predict(p_final)

YPRED_T=Y_std*YPRED_T+Y_mean 
YPRED_V=Y_std*YPRED_V+Y_mean
YPRED_TEST=Y_std*YPRED_TEST+Y_mean
#------------------------
#GENERATE PLOTS
#------------------------

##############################################################
####################### Visualization ########################
##############################################################

def plot_0():
    fig, ax = plt.subplots()
    ax.plot(epochs, loss_train, 'o', label='Training loss')
    ax.plot(epochs, loss_val, 'o', label='Validation loss')
    plt.xlabel('epochs', fontsize=18)
    plt.ylabel('loss', fontsize=18)
    plt.title("Training/Validation Loss Plot with "+ str(X.shape[1])+ " features")
    plt.legend(['Train Loss',"Validation Loss"])
    plt.show()
    
def plot_1(feature_num,xla='x',yla='y'):
    i = feature_num-1
    fig, ax = plt.subplots()
    ax.plot(X[train_idx,i]    , Y[train_idx],'o', label='Training') 
    ax.plot(X[val_idx,i]      , Y[val_idx],'x', label='Validation') 
    ax.plot(X[test_idx,i]     , Y[test_idx],'*', label='Test') 
    ax.plot(X[train_idx,i]    , YPRED_T,'.', label='Model') 
    plt.xlabel(X_keys[i], fontsize=18)
    plt.ylabel(Y_keys[0], fontsize=18)
    plt.legend(['Train','Validation','Test','Prediction'])
    plt.title("")
    plt.show()

def plot_2(xla='y_data',yla='y_predict'):
    fig, ax = plt.subplots()
    ax.plot(Y[train_idx]  , YPRED_T,'*', label='Training') 
    ax.plot(Y[val_idx]    , YPRED_V,'*', label='Validation') 
    ax.plot(Y[test_idx]    , YPRED_TEST,'*', label='Test') 
    plt.xlabel(xla, fontsize=18);	plt.ylabel(yla, fontsize=18); 	plt.legend()
    plt.title("True/Prediction Comparasion Plot with "+ str(X.shape[1])+ " features")
    plt.show()

if(IPLOT):
    plot_0()
    for num in range(X.shape[1]):
        i = num+1
        plot_1(i)
    plot_2()
