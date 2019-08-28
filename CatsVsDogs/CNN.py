#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Lalit Pal

A CNN model with 2 convolution layers and following maxpooling layers to solve
image classification problem. We will train our model on resized images of size 64x64 to reduce run time as we are training on CPU using Tensorflow 2.0 backend.

Dataset: https://www.kaggle.com/tongpython/cat-and-dog/downloads/cat-and-dog.zip/1

"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Activation, Flatten, Dense, Conv2D, MaxPooling2D

TRAIN_DATA_DIR = '/training_set'
TEST_DATA_DIR = '/test_set'
CNN_WEIGHTS_DOGS_VS_CATS = '/catvsdog_cnn_model.h5'
TARGET_IMAGE = '/cat.4003.jpg'

imgw = 64
imgh = 64
nb_train_samples = 8000
nb_test_samples = 2000
batch_size = 40
epochs = 25


def image_to_array(path):
    img = load_img(path)
    x = img_to_array(img)
    x = x.reshape((1,)+x.shape)
    return x

def create_model(input_shape):
    
    model = Sequential()
    
    model.add(Conv2D(32,(3,3), input_shape = input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(32,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(64,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten()) #flatter featue tensor to 1D
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.compile(loss = 'binary_crossentropy',
                  optimizer = 'rmsprop',
                  metric = ['accuracy'])
    
    return model

def train_model():
        
    datagen = ImageDataGenerator(rescale=1./255)
    
    model = create_model((64,64,3))
    #print(model)
    
    train_generator = datagen.flow_from_directory(
            TRAIN_DATA_DIR,
            target_size = (imgw, imgh),
            batch_size = batch_size,
            class_mode = 'binary',
            shuffle = False)
    
    test_generator = datagen.flow_from_directory(
            TEST_DATA_DIR,
            target_size = (imgw, imgh),
            batch_size = batch_size,
            class_mode = 'binary',
            shuffle = False)
    
    model.fit_generator(
            train_generator,
            steps_per_epoch = nb_train_samples, 
            epochs = epochs,
            validation_data = test_generator,
            validation_steps = nb_test_samples
            )
    
    model.save_weights(CNN_WEIGHTS_DOGS_VS_CATS)

def predict_model(model_path = CNN_WEIGHTS_DOGS_VS_CATS, img_path = TARGET_IMAGE):
    img = load_img(img_path, target_size = (imgw, imgh))
    img = img_to_array(img)/255
    img = img.reshape((1,) + img.shape)
    
    model = create_model((64,64,3))
    model.load_weights(CNN_WEIGHTS_DOGS_VS_CATS)
    
    print(model.predict(img))
    
train_model()
predict_model()   
