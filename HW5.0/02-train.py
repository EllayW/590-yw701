#!/usr/bin/env python
# coding: utf-8

# The model.fit and Train/Validation plot cells are commented due to the runtime.

# In[1]:


import pandas as pd
import numpy as np
from keras.optimizers import RMSprop
from keras.regularizers import l1, l2
from keras.preprocessing.text import Tokenizer
from keras.datasets import imdb
from keras import preprocessing
import random
from keras.models import Sequential 
from keras.layers import Dense, LSTM,Flatten
from keras.layers.embeddings import Embedding
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve,auc
#from keras.utils import to_categorical
from keras import layers 
from keras import models


# In[2]:


#################################################
## Hyper parameters:
rnn = {
    'num_epochs': 5, ## best n_epoch
    #batch_size = int(x_train.shape[0])
    'batch_size': 1, # mini-batch / batch / stochastic
    'learning_rate': .05,
    'LASSO' : l1(0.01),
    'RIDGE': l2(0.01),
    'loss': 'categorical_crossentropy',
    ## words..
    'max_features': 1000,
    'maxlen': 100,
    'n_words': 3000}

cnn = {
    'num_epochs':9,
    'batch_size': 1,
    'optimizer': 'rmsprop',
    'learning_rate': .01,
    'LASSO': l1(.01),
    'RIDGE': l2(.01),
    'metric': ['accuracy'],
    'loss': 'categorical_crossentropy',
    ## words..
    'max_features': 1000,
    'maxlen': 100,
    'n_words': 3000}


# In[3]:


## Read the data
df = pd.read_csv('processed_data .csv')
sentences = list(df.sentence)
labels = list(df.label)


# In[4]:


#################################################
## Tokenize the sentences by kera
tokenizer = Tokenizer(num_words=rnn['n_words'])
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
one_hot_results = tokenizer.texts_to_matrix(sentences, mode='binary')
# total len(sentences) number of sparse matrices
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


# In[5]:


######################################
### one-hot coder of y:
def to_one_hot(labels, dimension=3):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return (results)

## Now 'X':one_hot_results
ind_train = list(df[df.type == 'train'].index)
ind_val = list(df[df.type == 'val'].index)
ind_test = list(df[df.type == 'test'].index)

## Now 'Y': to_one_hot(y)
y_results = to_one_hot(list(df.label))

x_train = one_hot_results[ind_train]
y_train = y_results[ind_train]

x_val = one_hot_results[ind_val]
y_val = y_results[ind_val]

x_test = one_hot_results[ind_test]
y_test = y_results[ind_test]
true_labels = list(df.loc[ind_test,'label'])

x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=rnn['maxlen'])
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=rnn['maxlen'])
x_val = preprocessing.sequence.pad_sequences(x_val, maxlen=rnn['maxlen'])

#x_train_cnn = x_train.reshape((len(x_train),cnn['maxlen'],1))
#x_val_cnn = x_val.reshape((len(x_val),cnn['maxlen'],1))
#x_test_cnn = x_test.reshape((len(x_test),cnn['maxlen'],1))


# In[6]:


print('######################################')
print('Train/Validation/Test Split')
print('X_train','is',len(x_train))
print('X_val','is',len(x_val))
print('X_test','is',len(x_test))


# In[7]:


##########################################
## Modeling:
def RNN_model(regulation = None):
    model = Sequential()
    #learn 32-dimensional embeddings for each of the 20 words
    model.add(Embedding(rnn['max_features'], 32, input_length=rnn['maxlen']))
    model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(3, 
                    activation='softmax',
                    kernel_regularizer = regulation,))
    model.compile(optimizer='rmsprop',
                  loss=rnn['loss'], 
                  metrics=['acc'])
    
    return model

def CNN_model(regulation = None):
    model = models.Sequential()
    model.add(Embedding(rnn['max_features'], 32, input_length=rnn['maxlen']))
    model.add(layers.Conv1D(128, 3, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))
    
    model.compile(optimizer=cnn['optimizer'],
                      loss=cnn['loss'],
                      metrics=cnn['metric'])
    #model.summary()
    return(model) 
rnn_model = RNN_model(rnn['RIDGE'])
cnn_model = CNN_model(cnn['LASSO'])


# In[8]:


####################################
print('RNN model summary')
rnn_model.summary()


# In[9]:


####################################
print('CNN model summary')
cnn_model.summary()


# In[24]:


######################################
## Fit the model:

def hist(mod,name = 'rnn'):
    global history
    mod_name = eval(name)
    history = mod.fit(x_train,
                        y_train,
                        epochs=mod_name['num_epochs'],
                        batch_size=mod_name['batch_size'],
                        validation_data = (x_val,
                                           y_val))
    save_name = 'novels_'+name+'.h5'
    mod.save(save_name)
    return(history)

def visualization(history,name):
    file_name1 = 'train_val_loss_'+name+'.png'
    title1 ='Training and Validation Loss by '+name 
    
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    try:
        acc_values = history_dict['acc']
        val_acc_values = history_dict['val_acc']
    except:
        acc_values = history_dict['accuracy']
        val_acc_values = history_dict['val_accuracy']
    ### visualize the validation/train loss
    epochs = range(1, eval(name)['num_epochs'] + 1)
    plt.plot(epochs, loss_values, 'g', label='Training Loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation Loss')
    plt.title(title1)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    file_name2 = 'train_val_acc_'+name+'.png'
    title2 ='Training and Validation Accuracy by '+name 
    ### visualize the validation/train accuracy
    epochs = range(1, eval(name)['num_epochs'] + 1)
    plt.plot(epochs, acc_values, 'g', label='Training Accuracy')
    plt.plot(epochs, val_acc_values, 'b', label='Validation Accuracy')
    plt.title(title2)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


# In[26]:


############################
### RNN model
rnn_mod = hist(rnn_model)


# In[27]:


cnn_mod = hist(cnn_model,name = 'cnn')


# In[32]:


##########################################
## visualization
rnn_ = models.load_model('novels_Rnn.h5')
cnn_ = models.load_model('novels_cnn.h5')
print('###################################')
print('Train/Validation Metrics by RNN')
#visualization(rnn_mod,'rnn')


# In[33]:


##########################################
## visualization with CNN
print('###################################')
print('Train/Validation Metrics by CNN')
#visualization(cnn_mod,'cnn')


# In[15]:


eva_rnn = rnn_.evaluate(x_test,y_test)
print('########################')
print('Use Rnn, the loss is',eva_rnn[0])
print('the accuracy is',eva_rnn[1])


# In[16]:


eva_cnn = cnn_.evaluate(x_test,y_test)
print('########################')
print('Use Cnn, the loss is',eva_cnn[0])
print('the accuracy is',eva_cnn[1])


# In[17]:


## Prediction
print("########################")
print('ROC and AUC')

rnn_preds = rnn_.predict(x_test)
cnn_preds = cnn_.predict(x_test)


# In[18]:


dic0 = {0:'Monday or Tuesday',1:'Practical Agitation', 2:'Pride and Prejudice'}
def ROC(name = 'rnn'):
    
    print('#########################')
    print(name.upper()+'ROC')
    pred = name+"_preds"
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], eval(pred)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    colors = ['blue', 'red', 'green']
    for i, color in zip(range(3), colors):
        plt.plot(fpr[i], tpr[i], color=color,
                 label='title:{0} (area = {1:0.2f})'
                 ''.format(dic0[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for multi-class by '+name.upper())
    plt.legend(loc="lower right")
    plt.show()
    
    file_name = name+'_roc.png'


# In[19]:


ROC()


# In[20]:


ROC('cnn')


# In[31]:


### RUN this only in jupyter notebook
os.system(f'jupyter nbconvert 02-train.ipynb --to python')


# In[ ]:





# In[ ]:




