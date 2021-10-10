import os, shutil
import requests
from keras import layers
from keras import optimizers
from keras import models
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications import VGG16
from keras import backend as K
#import cv2
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
from keras.models import load_model
import cv2
#----------------------------------------
# Read the images from the train dataset
#----------------------------------------
#create 6 new folders
original_dataset_dir = os.path.join(os.getcwd(), 'train')
base_dir = os.path.join(os.getcwd(),'cats_and_dogs_small')

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')

#------------------------------------
# create base direction
#------------------------------------
# Since they are created, comment all such code to reduce the time cost
'''
os.mkdir(base_dir)

#------------------------------------
# create train/validation/test direction
#------------------------------------

os.mkdir(train_dir)
os.mkdir(validation_dir)
os.mkdir(test_dir)

#------------------------------------
# create cat/dog directions in train/validation/test direction
#------------------------------------

os.mkdir(train_cats_dir)
os.mkdir(train_dogs_dir)
os.mkdir(validation_cats_dir)
os.mkdir(validation_dogs_dir)
os.mkdir(test_cats_dir)
os.mkdir(test_dogs_dir)

#print(os.listdir(base_dir))


#------------------------------------
# place 1000/500/500 cat/dog pics into train/validation/test direction
#------------------------------------
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    #dst = train_cats_dir
    shutil.copy2(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)
    
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)
'''

#----------------------------------------
# CNN model with data augmentation,dropout
#----------------------------------------
def final_model():
	model = models.Sequential()
	model.add(layers.Conv2D(32, (3, 3), activation='relu',
	input_shape=(150, 150, 3)))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(64, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))

	model.add(layers.Conv2D(128, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(128, (3, 3), activation='relu'))

	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Flatten())
	model.add(layers.Dropout(0.5))

	model.add(layers.Dense(512, activation='relu'))
	model.add(layers.Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy',
	optimizer=optimizers.RMSprop(lr=1e-4),
	metrics=['acc'])

	model.compile(loss='binary_crossentropy',
		      optimizer=optimizers.RMSprop(lr=1e-4),
		      metrics=['acc'])
		     
	return(model)
	
#----------------------------------------
# Add generator for data augmentation
#----------------------------------------
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),
                                                    batch_size=20,
                                                    class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        target_size=(150, 150),
                                                        batch_size=20,
                                                        class_mode='binary')
 
# make sure test labels do not contain the augmentations

# 
def vis_plots():
#--------------------------
# Visualize the accuracy/loss in validation/train
#--------------------------

	acc = history.history['acc']
	val_acc = history.history['val_acc']

	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs = range(1, len(acc) + 1)

	plt.plot(epochs, acc, 'bo', label='Training acc')
	plt.plot(epochs, val_acc, 'b', label='Validation acc')
	plt.title('Training and validation accuracy')

	plt.legend()
	plt.figure()
	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.title('Training and validation loss')
	plt.legend()
	plt.show()                      

if __name__ == "__main__":
	
	#--------------------------
	# Save the model
	#--------------------------
	model = final_model()


		                                                
	history = model.fit_generator(train_generator,
		                      steps_per_epoch=20,
		                      epochs=20,
		                      validation_data=validation_generator,
		                      validation_steps=50) 
	vis_plots()
	model.save('cats_and_dogs_small.h5')   
	
	# Read the saved model
	model = load_model('cats_and_dogs_small.h5')
        # visualize the loss plots                      
	        
	#--------------------------
	# Visualize the activities in each layer
	#--------------------------

	## Choose the 1690th cat pic to test the model
	img_path = os.path.join(test_cats_dir, 'cat.1690.jpg')

	img = image.load_img(img_path, target_size=(150, 150))
	img_tensor = image.img_to_array(img)
	img_tensor = np.expand_dims(img_tensor, axis=0)
	img_tensor /= 255.

	plt.imshow(img_tensor[0])
	plt.title('Sample Cat Picture')
	plt.show()
	
	# now show all the activations in all the layers (first 8)
	
	layer_outputs = [layer.output for layer in model.layers[:8]]
	activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
	activations = activation_model.predict(img_tensor)      
	
	layer_names = []
	for layer in model.layers[:8]:
	    layer_names.append(layer.name)

	images_per_row = 16 

	for layer_name, layer_activation in zip(layer_names, activations):
	    n_features = layer_activation.shape[-1]
	    size = layer_activation.shape[1]

	    n_cols = n_features // images_per_row
	    display_grid = np.zeros((size * n_cols, images_per_row * size))
	    for col in range(n_cols):
	    	for row in range(images_per_row):
	    		# normalize the image
	    		channel_image = layer_activation[0,:, :,col * images_per_row + row]
		    	channel_image -= channel_image.mean()
		    	channel_image /= channel_image.std()
		    	channel_image *= 64
		    	channel_image += 128
		    	channel_image = np.clip(channel_image, 0, 255).astype('uint8')
		    	display_grid[col * size : (col + 1) * size,row * size : (row + 1) * size] =\
		    	channel_image
	    scale = 1. / size
	    plt.figure(figsize=(scale * display_grid.shape[1],
	    scale * display_grid.shape[0]))
	    plt.title(layer_name)

	    plt.grid(False)

	    plt.imshow(display_grid, aspect='auto', cmap='viridis')        
	    plt.show()   
	    
	#--------------------------
	# Visualizing convnet filters
	#--------------------------
	# check the heatmap of one layer (last layer)
	model2 = VGG16(weights='imagenet',include_top=False)
	layer_name = 'block3_conv1'
	filter_index = 0
	layer_output = model2.get_layer(layer_name).output
	loss = K.mean(layer_output[:, :, :, filter_index])	
	
	# get loss/gradient value
	grads = K.gradients(loss, model.input)[0]
	grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
	iterate = K.function([model.input], [loss, grads])
	loss_value, grads_value = iterate([np.zeros((1, 150, 150, 3))])

	input_img_data = np.random.random((1, 150, 150, 3)) * 20 + 128.
	step = 1.
	for i in range(40):
	    loss_value, grads_value = iterate([input_img_data])
	    input_img_data += grads_value * step      
	grads = K.gradients(loss, model.input)[0]
	grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
	iterate = K.function([model.input], [loss, grads])
	loss_value, grads_value = iterate([np.zeros((1, 150, 150, 3))])

	input_img_data = np.random.random((1, 150, 150, 3)) * 20 + 128.
	step = 1.
	for i in range(40):
	    loss_value, grads_value = iterate([input_img_data])
	    input_img_data += grads_value * step
	    
	# normalization to range(0,1)
	def deprocess_image(x):
	    x -= x.mean()
	    x /= (x.std() + 1e-5)
	    x *= 0.1 
	    x += 0.5 
	    x = np.clip(x, 0, 1)
	    x *= 255 
	    x = np.clip(x, 0, 255).astype('uint8')
	    return x

	# all the patterns
	def generate_pattern(layer_name, filter_index, size=150):
	    layer_output = model.get_layer(layer_name).output

	    loss = K.mean(layer_output[:, :, :, filter_index])
	    grads = K.gradients(loss, model.input)[0]

	    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

	    iterate = K.function([model.input], [loss, grads])

	    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.

	    step = 1.
	    for i in range(40):
	    	loss_value, grads_value = iterate([input_img_data])
	    	input_img_data += grads_value * step
	    	img = input_img_data[0]
	    return deprocess_image(img)
	    
	layer_name = 'block1_conv1'
	size = 64

	margin = 5
	results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))
	for i in range(8):
	    for j in range(8):
	    	filter_img = generate_pattern(layer_name, i + (j * 8), size=size)
	    	horizontal_start = i * size + i * margin
	    	horizontal_end = horizontal_start + size
	    	vertical_start = j * size + j * margin
	    	vertical_end = vertical_start + size
	    	results[horizontal_start: horizontal_end,vertical_start: vertical_end, :] = filter_img
	plt.figure(figsize=(20, 20))
	plt.imshow(results)
	plt.show()
	
	#--------------------------------------------
	# Visualize the heatmap
	#--------------------------------------------
	
	img_path = 'Meilong.jpg' # This is a picture of my dragon-lee cat (Chinese cat)
	img = image.load_img(img_path, target_size=(224, 224)) # reshape the picture to 224 x 224
	x = image.img_to_array(img) # covert it to arrays
	x = np.expand_dims(x, axis=0) 
	# now it is a 3-dimensional array, add another layer in the beginning
	
	x = preprocess_input(x)

	preds = model3.predict(x)# make a prediction
	print('Predicted:', decode_predictions(preds, top=3)[0])
	# it identifies my cat as a tabby/tiger/Egyptian cat, very accurate...
	
	np.argmax(preds[0]) 
	# the 281 the entries is recognized as the part that looks like a tabby the most
	# pull the last convolutional layer of the model with the 'tabby' entry 
	tabby_output = model3.output[:, 281]
	last_conv_layer = model3.get_layer('block5_conv3')

	# Gradient of the tabby class with regard to the output feature map of block5_conv3
	grads = K.gradients(tabby_output, last_conv_layer.output)[0]

	#Vector of shape (512,), where each entry is 
	# the mean intensity of the gradientover a specific feature-map channel
	pooled_grads = K.mean(grads, axis=(0, 1, 2))

	# Multiplies each channel in the feature-map array by “how important this
	# channel is” with regard to the“elephant” class
	iterate = K.function([model3.input],[pooled_grads, last_conv_layer.output[0]])
	pooled_grads_value, conv_layer_output_value = iterate([x])

	# the resulting feature map is the heatmap of the class activation
	for i in range(512):
	    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
	    heatmap = np.mean(conv_layer_output_value, axis=-1)
	    
	# visualize the normalized heatmap
	heatmap = np.maximum(heatmap, 0)
	heatmap /= np.max(heatmap)
	plt.matshow(heatmap)
	plt.show()
	
	

	# read and add the heatmap on the original cat pic
	img = cv2.imread(img_path)
	heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
	heatmap = np.uint8(255 * heatmap) # convert the heatmap to RGB
	heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
	superimposed_img = heatmap * 0.4 + img
	cv2.imwrite('Meilong_hm.jpg', superimposed_img)
