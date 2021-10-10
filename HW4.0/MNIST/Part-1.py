import skimage
import numpy as np
import matplotlib.pyplot as plt
from keras import layers 
from keras import models
import warnings
# datasets
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.regularizers import l1, l2
#from sklearn.model_selection import KFold
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras import models
warnings.filterwarnings("ignore")

print('This file might run for more than 10 mins')
#-------------------------------------
# Hyper Parameters
#-------------------------------------
f1=lambda NKEEP: int(0.01*NKEEP)
epochs=20
optimizer = 'rmsprop'
learning_rate = .01
LASSO = l1(learning_rate)
RIDGE = l2(learning_rate)
metric = ['accuracy']
loss = 'categorical_crossentropy'
k = 5
regulation = LASSO
model_name = 'CNN'

#-------------------------------------
# CNN Model
#-------------------------------------
def CNN():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', \
                            kernel_regularizer = regulation,\
                            input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    #model.summary()
    return(model)
    
#-------------------------------------
# DFF Model
#-------------------------------------
def DFF():
    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    model.add(layers.Dense(10,  activation='softmax'))
    #model.summary()
    return(model)
    
#-----------------------------
# Choose one Dataset:
#-----------------------------

def choose_dataset(name):
    if name not in ['mnist','fashion','cifar10']:
        name = 'mnist'
    #name = input('Please enter the dataset name (mnist, fashion, or cifar10): ')
    if name == 'mnist':
        #-------------------------------------
        #MNIST Dataset
        #-------------------------------------

        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        train_images = train_images.reshape((60000, 28, 28, 1))
        test_images = test_images.reshape((10000, 28, 28, 1))
    elif name == 'fashion':
        #-------------------------------------
        #MNIST FASHION Dataset
        #-------------------------------------

        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        train_images = train_images.reshape((60000, 28, 28, 1))
        test_images = test_images.reshape((10000, 28, 28, 1))
    elif name == 'cifar10':
        #-------------------------------------
        #CIFAR-10 Dataset
        #-------------------------------------

        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
        train_images.shape
        train_images = train_images.reshape((50000, 32,32,3))
        test_images = test_images.reshape((10000, 32,32, 3))
        
    #NORMALIZE
    train_images = train_images.astype('float32') / 255 
    test_images = test_images.astype('float32') / 255  
    
    #Categorize the labels
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    
    if model_name == 'DFF':
        pixel = train_images.shape[1]
        train_images = train_images.reshape((len(train_images), pixel * pixel))
        test_images = test_images.reshape((len(test_images), pixel * pixel))
        
    return((train_images, train_labels), (test_images, test_labels))
#-------------------------------------
#COMPILE MODEL
#-------------------------------------
def final_model():
    #model_name = input("Which model(CNN,DFF)?")
    model = eval(model_name)()
    model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metric)
    #model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size)
    return(model)
    
#-----------------------
# Evaluate a model
#-----------------------

def evaluation(X, y, n_folds=5,vis=False,model_name = 'CNN'):
    train_acc_histories = []
    val_acc_histories = []
    num_val_samples = len(X) // n_folds
    #kfold = KFold(n_folds, shuffle=True, random_state=8281)
    for i in range(k):
        print('The',i+1,' iteration is processing ---')
        #print('processing fold #', i)
        validation_images = X[i * num_val_samples: (i + 1) * num_val_samples]
        validation_labels = y[i * num_val_samples: (i + 1) * num_val_samples]
        train_images = np.concatenate([X[:i * num_val_samples],\
            X[(i + 1) * num_val_samples:]],axis=0)
        train_labels = np.concatenate([y[:i * num_val_samples],\
        y[(i + 1) * num_val_samples:]],axis=0)
        
        model = final_model()
        
        #-----------------------
        # Add Augmentation
        #-----------------------
        if model_name == 'CNN':
            train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=40,\
                width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,\
                zoom_range=0.2,horizontal_flip=True,)

            validation_datagen = ImageDataGenerator(rescale=1./255)

            train_datagen.fit(train_images)
            validation_datagen.fit(validation_images)

            train_generator = train_datagen.flow(train_images,train_labels,\
                batch_size=batch_size)

            validation_generator = validation_datagen.flow(\
                validation_images,validation_labels,batch_size=batch_size)

            history = model.fit_generator(generator=train_generator,\
                            validation_data=validation_generator, epochs=epochs, \
                                verbose=0)
        else:
            history = model.fit(train_images, train_labels,\
                validation_data=(validation_images, validation_labels),\
                    epochs=epochs, batch_size=batch_size, verbose=0)
        
        val_acc_history = history.history['val_accuracy']
        train_acc_history = history.history['accuracy']
        train_acc_histories.append(train_acc_history)
        val_acc_histories.append(val_acc_history)
        
    average_val_acc_history = [np.mean([x[i] for x in val_acc_histories]) \
            for i in range(epochs)]
    average_train_acc_history = [np.mean([x[i] for x in train_acc_histories]) \
            for i in range(epochs)]
    

    ## visualizations:
    if vis:
        plt.plot(range(1, len(average_val_acc_history) + 1), average_val_acc_history)
        plt.plot(range(1, len(average_train_acc_history) + 1),average_train_acc_history)
    return([average_val_acc_history,average_train_acc_history,model])
    
def visualize_layers():
##-----------------------------------------
# Visualize the original pic
#------------------------------------------
    if model_name == 'CNN':
        plt.imshow(test_images[8281], cmap=plt.get_cmap('gray'))
        plt.title("Random visualization in the chosen dataset")
        plt.show()

#------------------------------------
# Visualiza the intermediate activations
#------------------------------------
## choose any test image to predict its class and see the intermediate steps
        model.summary()

        layer_outputs = [layer.output for layer in model.layers[:5]]
        activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
        img_tensor = np.expand_dims(test_images[8281], axis=0)
        activations = activation_model.predict(img_tensor)
        layer_names = []

        # this model has 8 layers
        for layer in model.layers:
            layer_names.append(layer.name)
            images_per_row = 16

        for layer_name, layer_activation in zip(layer_names, activations):
            n_features = layer_activation.shape[-1]
            size = layer_activation.shape[1]

            n_cols = n_features // images_per_row
            display_grid = np.zeros((size * n_cols, images_per_row * size))
            for col in range(n_cols):
                for row in range(images_per_row):
                    channel_image = layer_activation[0,:, :,col * images_per_row + row]
                    channel_image -= channel_image.mean()

                    channel_image /= channel_image.std()

                    channel_image *= 64

                    channel_image += 128

                    channel_image = np.clip(channel_image, 0, 255).astype('uint8')

                    display_grid[col * size : (col + 1) * size,\
                                 row * size : (row + 1) * size] = channel_image
            scale = 1. / size

            plt.figure(figsize=(scale * display_grid.shape[1],
            scale * display_grid.shape[0]))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='gray')
            plt.show()
#--------------------------------
# Hyperparameter Tuning -- Best K
#--------------------------------

# the best performance is achieved when k = 5;
# however, it runs for too long. I skiped this step here.

#mod = []
#for k in range(2,6):
    #print(k)
#    mod = [i for i in evaluation(train_images, train_labels, n_folds=i,vis=False)[0]]
# k = mod.index(np.min(mod))
# k =5 here

#-----------------------------------
# Choose your dataset
#-----------------------------------

if __name__ == "__main__":
	(train_images, train_labels), (test_images, test_labels) = choose_dataset('mnist')
	NKEEP = int(train_images.shape[0]/10)
	batch_size = f1(NKEEP)
	'''
	### Debugging
	NKEEP = int(train_images.shape[0]/10)
	batch_size = f1(NKEEP)
	NKEEP=1000
	rand_indices = np.random.permutation(train_images.shape[0])
	if model_name == 'CNN':
	    train_images=train_images[rand_indices[0:NKEEP],:,:]
	elif model_name == 'DFF':
	    train_images=train_images[rand_indices[0:NKEEP],:]
	train_labels=train_labels[rand_indices[0:NKEEP]]
	'''

	#--------------------------------------------------------
	# Visualize validation / train accuracy with the best k
	#--------------------------------------------------------

	model = evaluation(train_images, train_labels, n_folds=5,vis=True,model_name = 'DFF')[2]
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.title("Validation/Train Accuracy when k=5")
	plt.legend(['validation', 'train'], loc='upper left')
	plt.show()

	#--------------------------------------
	# Save the model
	#--------------------------------------
	model.save('best_model.h5')

	#--------------------------------------
	# Load the model; Evaluate the model
	#--------------------------------------
	model = load_model('best_model.h5')
	_,test_acc = model.evaluate(test_images, test_labels, verbose=0)
	
	visualize_layers()

	print('The optimal model has the following parameters:')
	print('Kfold k = 5; '+ 'Epochs = 20')
	print('Regulation = LASSO')
	print('Batch Size = ',str(batch_size))
	print('The test accuracy is','%.3f' % (test_acc * 100))
	print('-------')
