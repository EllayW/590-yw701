#!/usr/bin/env python
# coding: utf-8

# In[18]:


import os
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import MaxPooling2D, UpSampling2D
from keras.models import Sequential,Model
from keras.layers import Dense, Activation
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10, cifar100
from keras.models import load_model


# In[19]:


################################################################
####################### Training ###############################
################################################################


# In[20]:


### define hyperparameters:
epochs=20
optimizer = 'rmsprop'
loss = 'mean_squared_error'
batch_size=100


# In[27]:


### read the input:
(X, Y), (test_images, test_labels) = cifar10.load_data()


# In[28]:


### normalization:
X = X.astype('float32') / 255.
test_images = test_images.astype('float32') / 255.


### flattening:
X = X.reshape((len(X), 32,32,3))
test_images = test_images.reshape((len(test_images), 32,32,3))


# In[37]:


### build the base model:
def ConV():
    input_img = Input(shape=(32, 32, 3))
    enc_conv1 = Conv2D(12, (3, 3), activation='relu', padding='same')(input_img)
    enc_pool1 = MaxPooling2D((2, 2), padding='same')(enc_conv1)
    enc_conv2 = Conv2D(8, (4, 4), activation='relu', padding='same')(enc_pool1)
    enc_ouput = MaxPooling2D((4, 4), padding='same')(enc_conv2)

    dec_conv2 = Conv2D(8, (4, 4), activation='relu', padding='same')(enc_ouput)
    dec_upsample2 = UpSampling2D((4, 4))(dec_conv2)
    #dec_conv3 = Conv2D(12, (3, 3), activation='relu')(dec_upsample2)
    dec_upsample3 = UpSampling2D((2, 2))(dec_upsample2)
    dec_output = Conv2D(3, (3, 3), activation='linear', padding='same')(dec_upsample3)

    autoencoder = Model(input_img, dec_output)
    autoencoder.compile(optimizer='rmsprop', loss=loss)
    autoencoder.summary()
    return(autoencoder)

### evaluate the model:
def evaluation(mod,X,vis=True):
    global train_mse_histories
    global val_mse_histories
    
    train_mse_histories = []
    val_mse_histories = []
    
    history = mod.fit(X, X,validation_split=0.2,
        epochs=epochs, batch_size=batch_size)
    
    val_mse_history = history.history['val_loss']
    train_mse_history = history.history['loss']
    train_mse_histories.append(train_mse_history)
    val_mse_histories.append(val_mse_history)
        
    average_val_mse_history = [np.mean([x[i] for x in val_mse_histories])             for i in range(epochs)]
    average_train_mse_history = [np.mean([x[i] for x in train_mse_histories])             for i in range(epochs)]
    
    save_name = 'HW6.3.h5'
    mod.save(save_name)
    ## visualizations:
    if vis:
        plt.plot(range(1, len(average_val_mse_history) + 1), average_val_mse_history)
        plt.plot(range(1, len(average_train_mse_history) + 1),average_train_mse_history)
    return([average_val_mse_history,average_train_mse_history])


# In[38]:


### Build the model:
model = ConV()
avg_val,avg_train = evaluation(model,X)
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.title("Validation/Train MSE")
plt.legend(['validation', 'train'], loc='upper left')
plt.show()


# In[39]:


### print out the model summary:
plt.savefig('HW6.3-train-val-loss.png')
model.summary()
model = load_model('HW6.3.h5')
model.evaluate(X,X,batch_size = batch_size)


# In[13]:


model.evaluate(test_images,test_images,batch_size = batch_size)


# In[56]:


################################################################
####################### Fashion Data ###########################
################################################################


# In[40]:


(X_cifar100, Y_cifar100), (test_images, test_labels) = cifar100.load_data()


# In[55]:


### remove the truck
l0 = [i for i in range(len(Y_cifar100)) if Y_cifar100[i] != 58]
l1 = [i for i in range(len(test_labels)) if test_labels[i] != 58]
X2 = X_cifar100[l0]
test_images = test_images[l1]


# In[53]:


X2.shape


# In[56]:


### normalization:
X2 = X2.astype('float32') / 255.
test_images = test_images.astype('float32') / 255.

### flattening:
X2 = X2.reshape((len(X2), 32,32,3))


# In[63]:


### visualize the input/output:
X3=model.predict(test_images)
X4 = model.predict(X2)
test_images=test_images.reshape(9900,32,32,3)
X3=X3.reshape(9900,32,32,3)
X4=X4.reshape(49500,32,32,3)


# In[64]:


f, ax = plt.subplots(6,1)
I1=1
I2=2
ax[0].imshow(test_images[I1])
ax[1].imshow(X3[I1])
ax[2].imshow(test_images[I2])
ax[3].imshow(X3[I2])
ax[4].imshow(X4[I1])
ax[5].imshow(X4[I2])
plt.show()


# In[65]:


f.savefig('HW6.2-history.png')


# In[66]:


### define threshold:
thres = 4*model.evaluate(X,X,batch_size = batch_size)


# In[67]:


def count_anomaly(dataset):
    ## dataset = X_fas or X
    count = 0
    for i in range(dataset.shape[0]):
        dat = dataset[i].reshape(1,32,32,3)
        err = model.evaluate(dat,dat)
        if err > thres:
            count +=1
    return(count)


# In[68]:


### count anomalies in the train dataset
count1 = count_anomaly(X)
print(count1)


# In[69]:


### count anomalies in the anomaly dataset
count2 = count_anomaly(X2)


# In[70]:


count2


# In[ ]:




