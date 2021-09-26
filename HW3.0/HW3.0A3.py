from keras.datasets import reuters
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import models
from keras import layers
import numpy as np
#from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from keras.optimizers import RMSprop
from keras.regularizers import l1, l2
from sklearn.metrics import confusion_matrix
import seaborn as sns

(train_data, train_labels), (X_test, y_test) = reuters.load_data(num_words=10000)
X_train, X_val, y_train, y_val = train_test_split(train_data,train_labels,
                                                                    test_size = 0.2, 
                                                                    random_state = 444)
print(len(X_train))
print(len(X_val))
print(len(X_test))

#############################################################################
########################### No Normalization Needed #########################
#############################################################################

word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_newswire = ' '. join([reverse_word_index.get(i - 3, '?') \
                              for i in X_train[0]])
                              
### split train,test,validation set
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(X_train)
x_val = vectorize_sequences(X_val)
x_test = vectorize_sequences(X_test)

### one-hot coder:
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

one_hot_train_labels = to_one_hot(y_train)
one_hot_val_labels = to_one_hot(y_val)
one_hot_test_labels = to_one_hot(y_test)

#############################################################################
############################ Hyperparameter Tuning ##########################
#############################################################################

num_epochs = 5
#batch_size = int(x_train.shape[0])
batch_size = 1 # mini-batch / batch / stochastic
learning_rate = .05
LASSO = l1(0.01)
RIDGE = l2(0.01)
loss = 'categorical_crossentropy'

def baseline_model(regulation = None):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(10000,)))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(46, activation='softmax')) # class dim = 46       
    opt = RMSprop(learning_rate=learning_rate)
    model.compile(optimizer=opt,
                  loss=loss,
                  metrics=['accuracy'])
    return model
    
model = baseline_model(RIDGE)
history = model.fit(x_train,
                    one_hot_train_labels,
                    epochs=num_epochs,
                    batch_size=batch_size,
                    validation_data=(x_val, one_hot_val_labels))
                    
#############################################################################
############################### Visualizations ##############################
#############################################################################
               
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']

### visualize the validation/train loss
epochs = range(1, num_epochs + 1)
plt.plot(epochs, loss_values, 'g', label='Training Loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#############################################################################
############################### Fit the Model ###############################
#############################################################################

model.fit(x_train, one_hot_train_labels, epochs=num_epochs, batch_size=batch_size)
pred = model.predict(x_test)
test_acc = model.evaluate(x_test, one_hot_test_labels)[1]

y_pred = [np.argmax(pred[i,:]) for i in range(len(pred))]
print('The confusion matrix is:')
print(confusion_matrix(y_test,y_pred, labels=range(1,46+1)))
conf = confusion_matrix(y_test,y_pred, labels=range(1,46+1))

#############################################################################
################################# Results ###################################
#############################################################################


print('The optimal model has the following parameters:')
print('Regulation = None')
print('Mini-Batch or Batch or Stochastic: Batch')
print('Epoch size = 5')
print('-------')
print('The training accuracy is '+str(np.round(val_acc_values[num_epochs-1],3)))
print('The validation accuracy is '+str(np.round(acc_values[num_epochs-1],3)))
print('The test accuracy is '+str(np.round(test_acc,3)))
