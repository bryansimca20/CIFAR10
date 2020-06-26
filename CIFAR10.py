#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 15:58:05 2020

@author: bryansimca20
"""

# IMPORT OF LIBRARIES

import tensorflow as tf

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.backend import repeat
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from tensorflow.keras.datasets import cifar10 #load the dataset
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

import numpy as np
import matplotlib.pyplot as plt

# PREPARATION OF THE DATA

# load train and test dataset
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = cifar10.load_data()
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY

# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')

	# normalize to range 0-1
    # Since the CIFAR 10 dataset is in RGB
    # dividing the pixels by 255 will normalize
    # it from 0 to 1 which is a grayscale format.
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0

	# return normalized images
	return train_norm, test_norm


# define cnn model
def define_model():
    weight_decay = 1e-4

    #sequential just means layer after layer
    model = Sequential()

    # 3 VGG Block Architecture Style:
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', kernel_regularizer = regularizers.l2(weight_decay), padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', kernel_regularizer = regularizers.l2(weight_decay), padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', kernel_regularizer = regularizers.l2(weight_decay), padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', kernel_regularizer = regularizers.l2(weight_decay), padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', kernel_regularizer = regularizers.l2(weight_decay), padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', kernel_regularizer = regularizers.l2(weight_decay), padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))

    # DENSE OUTPUT LAYER:
    # Cannot use sigmoid activation in the last dense layer (the results layer)
    # because it only outputs a binary from 0 to 1. use softmax instead.
    model.add(Dense(10, activation='softmax'))

    model.summary()
	# compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# plot diagnostic learning curves
def plot_diagnostics(history):
 	# plot loss
    plt.figure(1)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')
 	# plot accuracy
    plt.figure(2)
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')


# run the test harness for evaluating a model
def run_test_harness():
	# load dataset
    trainX, trainY, testX, testY = load_dataset()
	# prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
	# define model
    model = define_model()

    # DATA AUGMENTATION:
    # making more copies of the dataset with small modifications
    # create data generator
    datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
	# prepare iterator
    it_train = datagen.flow(trainX, trainY, batch_size=64)
	# fit model
    steps = int(trainX.shape[0] / 64)

	# fit model
    history = model.fit_generator(it_train, steps_per_epoch=steps, epochs=200, validation_data=(testX, testY), verbose=1)

	# evaluate model
    _, acc = model.evaluate(testX, testY, verbose=0)
    print('> %.3f' % (acc * 100.0))

    # learning curves
    plot_diagnostics(history)
    classes_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    #plot confusion matrix
    predY = model.predict_classes(testX)
    rounded_labels = np.argmax(testY, axis = 1)
    mat = confusion_matrix(rounded_labels, predY)
    plot_confusion_matrix(mat, figsize=(9,9), show_normed = True, class_names = classes_name)

    model.save('CIFAR10_2DENSELAYER_MODEL.h5')

# entry point, run the test harness
run_test_harness()






