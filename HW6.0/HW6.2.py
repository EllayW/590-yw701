#!/usr/bin/env python
# coding: utf-8

# In[59]:


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
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras.models import load_model


# In[60]:


################################################################
####################### Training ###############################
################################################################


# In[61]:


### define hyperparameters:
epochs=20
optimizer = 'rmsprop'
loss = 'mean_squared_error'
batch_size=100


# In[62]:


### read the input:
(X, Y), (test_images, test_labels) = mnist.load_data()


# In[63]:


### normalization:
X = X.astype('float32') / 255.
test_images = test_images.astype('float32') / 255.

### flattening:
X = X.reshape((len(X), 28,28,1))
test_images = test_images.reshape((len(test_images), 28,28,1))


# In[64]:


### build the base model:
def ConV():
    input_img = Input(shape=(28, 28, 1))
    enc_conv1 = Conv2D(10, (3, 3), activation='relu', padding='same')(input_img)
    enc_pool1 = MaxPooling2D((2, 2), padding='same')(enc_conv1)
    enc_conv2 = Conv2D(6, (4, 4), activation='relu', padding='same')(enc_pool1)
    enc_ouput = MaxPooling2D((4, 4), padding='same')(enc_conv2)

    dec_conv2 = Conv2D(6, (4, 4), activation='relu', padding='same')(enc_ouput)
    dec_upsample2 = UpSampling2D((4, 4))(dec_conv2)
    dec_conv3 = Conv2D(10, (3, 3), activation='relu')(dec_upsample2)
    dec_upsample3 = UpSampling2D((2, 2))(dec_conv3)
    dec_output = Conv2D(1, (3, 3), activation='linear', padding='same')(dec_upsample3)

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
    
    save_name = 'HW6.2-conv.h5'
    mod.save(save_name)
    ## visualizations:
    if vis:
        plt.plot(range(1, len(average_val_mse_history) + 1), average_val_mse_history)
        plt.plot(range(1, len(average_train_mse_history) + 1),average_train_mse_history)
    return([average_val_mse_history,average_train_mse_history])


# In[65]:


### Build the model:
model = ConV()
avg_val,avg_train = evaluation(model,X)
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.title("Validation/Train MSE")
plt.legend(['validation', 'train'], loc='upper left')
plt.show()


# In[66]:


### print out the model summary:
plt.savefig('HW6.2-train-val-loss.png')
model.summary()
model = load_model('HW6.2-conv.h5')
model.evaluate(X,X,batch_size = batch_size)


# In[67]:


model.evaluate(test_images,test_images,batch_size = batch_size)


# In[68]:


################################################################
####################### Fashion Data ###########################
################################################################


# In[69]:


(X_fas, Y_fas), (test_images_fas, test_labels_fas) = fashion_mnist.load_data()


# In[70]:


### normalization:
X_fas = X_fas.astype('float32') / 255.
#test_images_fas = test_images_fas.astype('float32') / 255.

### flattening:
X_fas = X_fas.reshape((len(X_fas), 28,28,1))


# In[71]:


### visualize the input/output:
X1=model.predict(test_images)
X2 = model.predict(X_fas)
test_images=test_images.reshape(10000,28,28)
X1=X1.reshape(10000,28,28)
X2=X2.reshape(60000,28,28)

f, ax = plt.subplots(6,1)
I1=1
I2=2
ax[0].imshow(test_images[I1])
ax[1].imshow(X1[I1])
ax[2].imshow(test_images[I2])
ax[3].imshow(X1[I2])
ax[4].imshow(X2[I1])
ax[5].imshow(X2[I2])
plt.show()


# In[72]:


f.savefig('HW6.2-history.png')


# In[73]:


### define threshold:
thres = 4*model.evaluate(X,X,batch_size = batch_size)


# In[74]:


def count_anomaly(dataset):
    ## dataset = X_fas or X
    count = 0
    for i in range(dataset.shape[0]):
        dat = dataset[i].reshape(1,28,28,1)
        err = model.evaluate(dat,dat)
        if err > thres:
            count +=1
    return(count)


# In[75]:


### count anomalies in the train dataset
count1 = count_anomaly(X)
print(count1)


# In[76]:


### count anomalies in the anomaly dataset
count2 = count_anomaly(X_fas)


# In[77]:


count2


# In[ ]:




