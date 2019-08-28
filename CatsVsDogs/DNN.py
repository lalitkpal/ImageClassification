#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Lalit Pal

Simple Deep Learning model with 2 hidden layer to classify cats and dogs.
We will train our model on resized images of size 64x64 to reduce run time as we are training on CPU using Tensorflow 2.0 backend.

Data source :https://www.kaggle.com/tongpython/cat-and-dog/downloads/cat-and-dog.zip/1

"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

import tensorflow as tf

TRAIN_DATA_DIR = '/training_set'
TEST_DATA_DIR = '/test_set'
DNN_WEIGHTS_DOGS_VS_CATS = '/catvsdog_dnn_model.h5'
TARGET_IMAGE = '/dog.4.jpg'

imgw = 64
imgh = 64
nb_train_samples = 8000
nb_test_samples = 2000
epochs = 25
batch_size = 40

def image_to_array(path):
    img = load_img(path)
    x = img_to_array(img)
    x = x.reshape((1,)+x.shape)
    return x


def create_model(input_shape):
    model = Sequential()
    
    model.add(Flatten(input_shape = input_shape))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('tanh'))
    
    model.compile(optimizer = 'rmsprop', 
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])
    
    return model

def train_model():
        
    datagen = ImageDataGenerator(rescale=1./255)
    
    model = create_model((64,64,3))
    print(model)
    
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
            validation_steps = nb_train_samples
            )
    
    model.save_weights(DNN_WEIGHTS_DOGS_VS_CATS)

def predict_model(model_path = DNN_WEIGHTS_DOGS_VS_CATS, img_path = TARGET_IMAGE):
    img = load_img(img_path, target_size = (imgw, imgh))
    img = img_to_array(img)/255
    img = img.reshape((1,) + img.shape)
    
    model = create_model((64,64,3))
    model.load_weights(DNN_WEIGHTS_DOGS_VS_CATS)
    
    print(model.predict(img))
    
train_model()
predict_model()   
