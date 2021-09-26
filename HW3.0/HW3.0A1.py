from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import models
from keras import layers
import numpy as np
#from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.regularizers import l1, l2

(train_data, train_targets), (test_data, test_targets) =boston_housing.load_data()

#print(len(train_data))

##########################################################################
########################### Normalization#################################
##########################################################################

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

##########################################################################
########################### Hyperparameter ###############################
##########################################################################

#activation = 'linear'
optimizer = 'rmsprop'
num_epochs = 46
batch_size = 1 # stochastic
learning_rate = .01
LASSO = l1(learning_rate)
RIDGE = l2(learning_rate)
loss = 'mse'
k = 5
regulation = LASSO

def baseline_model(regulation = None):
    model = Sequential()
    model.add(Dense(64, activation='relu',kernel_regularizer = regulation,
    input_shape=(train_data.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])
    return model

##########################################################################
################################# Kfold ##################################
##########################################################################

def bestk(k=4,vis = False,num_epochs = num_epochs,regulation = None):
    num_val_samples = len(train_data) // k

    global val_mse_histories,train_mse_histories
    val_mse_histories = []
    train_mse_histories = []
    val_mae_histories = []
    train_mae_histories = []
    for i in range(k):
        #print('processing fold #', i)
        val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
        partial_train_data = np.concatenate([train_data[:i * num_val_samples],\
            train_data[(i + 1) * num_val_samples:]],axis=0)
        partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],\
        train_targets[(i + 1) * num_val_samples:]],axis=0)
        model = baseline_model(LASSO)
        history = model.fit(partial_train_data, partial_train_targets,\
                validation_data=(val_data, val_targets),\
                    epochs=num_epochs, batch_size=1, verbose=0)
        val_mae_history = history.history['val_mae']
        train_mae_history = history.history['mae']
        val_mse_history = history.history['val_loss']
        train_mse_history = history.history['loss']
        
        val_mae_histories.append(val_mae_history)
        train_mae_histories.append(train_mae_history)
        val_mse_histories.append(val_mse_history)
        train_mse_histories.append(train_mse_history)
        
    average_val_mse_history = [np.mean([x[i] for x in val_mse_histories]) \
            for i in range(num_epochs)]
    average_train_mse_history = [np.mean([x[i] for x in train_mse_histories]) \
            for i in range(num_epochs)]
    average_val_mae_history = [np.mean([x[i] for x in val_mae_histories]) \
            for i in range(num_epochs)]
    average_train_mae_history = [np.mean([x[i] for x in train_mae_histories]) \
            for i in range(num_epochs)]

    ## visualizations:
    if vis:
        plt.plot(range(1, len(average_val_mse_history) + 1), average_val_mse_history)
        plt.plot(range(1, len(average_train_mse_history) + 1),average_train_mse_history)
    return([average_val_mse_history,average_train_mse_history,\
           average_val_mae_history,average_train_mae_history])
           
## Find the best k:
#dic0 ={}
#for k in range(2,6):
#for k in [5]:
    #print(k)
#    mod = bestk(k,False,100,LASSO)
#    dic0[k] = [np.min(mod),mod.index(np.min(mod))]
print('Choose k = 5, epoch = 46')

## visualize the validation / train / test mae:
scores = bestk(5,True,46)
#plt.axvline(x=46, color='k', linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("Kfold with k = "+str(k))
plt.legend(['validation', 'train'], loc='upper left')
plt.show()

train_mae_score = scores[3][46-1]
val_mae_score = scores[2][46-1]

##########################################################################
########################### Visualizations ###############################
##########################################################################

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

smooth_mae_history = smooth_curve(scores[2][10:])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

##########################################################################
######################## Fitting the model ###############################
##########################################################################

model = baseline_model(regulation)
model.fit(test_data, test_targets,epochs=num_epochs, batch_size=batch_size, verbose=0)
test_mae_score, test_mse_score = model.evaluate(test_data, test_targets)

pred = model.predict(test_data)

plt.scatter(test_targets,pred, color='g')
plt.xlabel('Test Target')
plt.ylabel("Prediction")
plt.title('Test Targets vs. Prediction by Linear Regression')
#plt.plot(train_data[:,0], pred, color='r')
plt.show()

############################################################################
################################ Results ###################################
############################################################################

print('The optimal model has the following parameters:')
print('Kfold k = 5; '+ 'Epochs = 46')
print('Regulation = LASSO')
print('Batch Size = 1 (stochastic)')
print('-------')
print('The training MAE is '+str(np.round(train_mae_score,3)))
print('The validation MAE is '+str(np.round(val_mae_score,3)))
print('The test MAE is '+str(np.round(test_mae_score,3)))

