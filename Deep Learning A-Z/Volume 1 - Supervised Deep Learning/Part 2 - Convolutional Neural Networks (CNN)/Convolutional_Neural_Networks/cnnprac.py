# -*- coding: utf-8 -*-
"""
Created on Thu May 18 07:32:06 2017

@author: SHRADDHA
"""

#Convolutional Neural Network

#Part 1 : Data Proprocessing -- Given the structure of folders, its not required

#Part 2 - Building the CNN

#Importing the libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initialization of CNN
classifier = Sequential()

#Addition of layers
#Step 1 Convolution
classifier.add(Convolution2D(32,(3,3),padding = 'same',
                             input_shape = (64,64,3), activation = 'relu'))

#Step 2 MaxPooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Step 3 Flattening
classifier.add(Flatten())

#Step 4 Full Connection
classifier.add(Dense(units = 128, activation = 'relu'))  #Input Layer
classifier.add(Dense(units = 1, activation = 'sigmoid')) #Output Layer

#Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                   metrics = ['accuracy'])



#Part 2 Fitting the CNN to images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')


classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)


#Part 3 Prediction Step
import numpy as np
from keras.preprocessing import image

#Load Image
test_image = image.load_image('dataset/single_prediction/cat_or_dog_1.jpg',
                              target_size = (64,64))
#Add new dimension to convert into 3D array
test_image = image.img_to_array(test_image)
#Add one dimension again as done during training
test_image = np.expand_dims(test_image, axis = 0)
classifier.predict(test_image)
training_set.class_indices

if(result[0][0] == 1):
    prediction = 'dog'
else:
    prediction = 'cat'

