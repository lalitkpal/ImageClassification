#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Lalit Pal

Image classification model based on VGG16 model. We will train our model on resized images of size 64x64 to reduce run time as we are training on CPU using Tensorflow 2.0 backend.

Dataset: Dataset: https://www.kaggle.com/tongpython/cat-and-dog/downloads/cat-and-dog.zip/1

"""

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras import applications


#dimensions of input images
img_width, img_height = 64,64

top_model_weights_path = "/vgg_16_extend.h5"
train_data_dir = '/training_set'
test_data_dir = '/test_set'
TARGET_IMAGES = '/validation'
TARGET_IMAGE = '/dog.4.jpg'

nb_train_samples = 8000
nb_test_samples = 2000
epochs = 25
batch_size = 40

def create_top_model(input_shape):
    
    model = Sequential()
    
    model.add(Flatten(input_shape = input_shape))
    model.add(Dense(256,activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1,activation = 'sigmoid'))
    
    model.compile(optimizer = 'rmsprop', 
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])
    
    return model

def save_bottleneck_features():
    datagen = ImageDataGenerator(rescale=1./255)
    
    #build vgg16 net
    model =  applications.VGG16(include_top = False, weights = 'imagenet')
    
    generator = datagen.flow_from_directory(
            train_data_dir,
            target_size = (img_width, img_height),
            batch_size = batch_size,
            class_mode = None,
            shuffle = False)
    #print(model)
    bottleneck_features_train = model.predict_generator(generator, nb_train_samples//batch_size)
    np.save(open('bottleneck_features_train.npy','wb'),
            bottleneck_features_train)
    
    generator = datagen.flow_from_directory(
            test_data_dir,
            target_size = (img_width, img_height),
            batch_size = batch_size,
            class_mode = None,
            shuffle = False)
    
    bottleneck_features_test = model.predict_generator(generator, nb_test_samples//batch_size)
    np.save(open('bottleneck_features_validation.npy','wb'),
            bottleneck_features_test)
    
    
def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy','rb'))
    train_labels = np.array([0]*4000 + [1]*4000)
    
    validation_data = np.load(open('bottleneck_features_validation.npy','rb'))
    validation_labels = np.array([0]*1000 + [1]*1000)
     
    model = create_top_model(train_data.shape[1:])
   
    model.fit(train_data, train_labels, epochs=epochs,
              batch_size=batch_size,
              validation_data = (validation_data,validation_labels))
    
    model.save_weights(top_model_weights_path)
    

def prediction_using_saved_model(model_path = top_model_weights_path, img_path = TARGET_IMAGES):
    img = load_img(img_path, target_size = (img_width, img_height))
    img = img_to_array(img)/255
    img = img.reshape((1,) + img.shape)
    
    #print(img.shape)
    vgg_model = applications.VGG16(include_top = False, weights = 'imagenet')
    
    img_features = vgg_model.predict(imgarr)
    
    model = create_top_model(img_features.shape[1:])
        
    model.load_weights(top_model_weights_path)
    
    img_features = img_features.reshape((1,)+img_features.shape)
    
    print(model.predict(img_features))
    
save_bottleneck_features()
train_top_model()
prediction_using_saved_model(TARGET_IMAGE)
