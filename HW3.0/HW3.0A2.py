from keras.datasets import imdb
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

(train_data, train_labels), (X_test, y_test) = imdb.load_data(num_words=10000)
X_train, X_val, y_train, y_val = train_test_split(train_data,train_labels,
                                                                    test_size = 0.2, 
                                                                    random_state = 44)
print(len(X_train))
print(len(X_val))
print(len(X_test))

############################################################################
########################### No Normalization Needed ########################
############################################################################
print('Since the train/validation/test dataset contains only 0s and 1s, we do not need to normalize it.')

word_index = imdb.get_word_index()
reverse_word_index = dict(
[(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join(
[reverse_word_index.get(i - 3, '?') for i in X_train[0]])

### split train,test,validation set
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results
x_train = vectorize_sequences(X_train)
x_val = vectorize_sequences(X_val)
x_test = vectorize_sequences(X_test)

y_train = np.asarray(y_train).astype('float32')
y_val = np.asarray(y_val).astype('float32')
y_test = np.asarray(y_test).astype('float32')

############################################################################
########################### Hyperparameter Tuning ##########################
############################################################################

num_epochs = 7
batch_size = int(x_train.shape[1]) # mini-batch / batch / stochastic
learning_rate = 5
LASSO = l1(0.01)
RIDGE = l2(0.01)
loss = 'binary_crossentropy'

def baseline_model(regulation = None):
    model = Sequential() 
    #model.add(Dense(16, activation='relu', input_shape=(10000,)))
    #model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    opt = RMSprop(learning_rate=learning_rate)
    model.compile(optimizer=opt,
                  loss=loss,
                  metrics=['accuracy'])
    return model

model = baseline_model()
history = model.fit(x_train,
                    y_train,
                    epochs=num_epochs,
                    batch_size=batch_size,
                    validation_data=(x_val, y_val))
                    
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']

############################################################################
############################# Visualizations ###############################
############################################################################

epochs = range(1, num_epochs + 1)
plt.plot(epochs, loss_values, 'g', label='Training Loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

############################################################################
############################## Fit the Model ###############################
############################################################################

model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size)
pred = model.predict(x_test)
test_acc = model.evaluate(x_test, y_test)[1]
plt.legend()
plt.show()

############################################################################
################################# Results ##################################
############################################################################

print('The optimal model has the following parameters:')
print('Regulation = None')
print('Mini-Batch or Batch or Stochastic: Batch')
print('Epoch size = 7')
print('-------')
print('The training accuracy is '+str(np.round(val_acc_values[num_epochs-1],3)))
print('The validation accuracy is '+str(np.round(acc_values[num_epochs-1],3)))
print('The test accuracy is '+str(np.round(test_acc,3)))

