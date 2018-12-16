from os import listdir
from os.path import isfile, join
from scipy import fftpack, ndimage
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import h5py
import keras
import tensorflow as tf
import os.path
import random
from keras.models import load_model
 
def readSet(dir):
    files = []
    for i, className in enumerate(sorted(listdir(dir))):
        if className == '.DS_Store':
            continue
        ind = classesToIds[className]
        classDir = dir + '/' + className
        for filename in listdir(classDir):
            path = classDir + '/' + filename
            files.append((path, ind))
    return getImages(files)

def getImages(paths):
    images = []
    for (path, classId) in paths:
        images.append((transformImage(ndimage.imread(path)), classId))
    return images

def transformImage(image):
    result_amplitude = np.zeros((100, 100, 6))
    for i in range(3):
        fft = np.fft.fft2(image[:, :, i])
        fshift = np.fft.fftshift(fft)
        result_amplitude[:, :, i] = np.abs(fshift) / 2445836.0
        result_amplitude[:, :, 3+i] = np.angle(fshift) / 3.14
    return result_amplitude


def getClassesIds():
    classesToIds = {}
    idsToClasses = {}
    for i, className in enumerate(listdir('fruits-360/Training')):
        if className == '.DS_Store':
            continue
        if className not in classesToIds:
            classesToIds[className] = i
            idsToClasses[i] = className
    return classesToIds, idsToClasses
 
classesToIdsFile = 'classesToIds'
idsToClassesFile = 'idsToClasses'
 
if not os.path.exists(classesToIdsFile):
    classesToIds, idsToClasses = getClassesIds()
    with open(classesToIdsFile, 'wb') as handle:
        pickle.dump(classesToIds, handle)
    with open(idsToClassesFile, 'wb') as handle:
        pickle.dump(idsToClasses, handle)
 
with open(classesToIdsFile, 'rb') as handle:
    classesToIds = pickle.load(handle)
 
with open(idsToClassesFile, 'rb') as handle:
    idsToClasses = pickle.load(handle)

def generate_arrays_from_files(files2):
    batch_features = np.zeros((batch_size, 100, 100, 6))
    batch_labels = np.zeros((batch_size, 1))
    while 1:
        for i in range(batch_size):
            index = random.choice(files2)
            file, clas = index
            batch_features[i] = file
            batch_labels[i] = clas
        yield (batch_features, batch_labels)

files = readSet('fruits-360/Training')
batch_size=100
samples = np.ceil(len(files) / batch_size)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(100, 100, 6)),
    keras.layers.Dense(2048, activation=tf.nn.relu),
    keras.layers.Dense(2048, activation=tf.nn.relu),
    keras.layers.Dense(2048, activation=tf.nn.relu),
    keras.layers.Dense(84, activation=tf.nn.softmax)
])
 
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
 
model.fit_generator(generate_arrays_from_files(files), samples_per_epoch=samples*2, epochs=2000)
model.save('my_model.h5')

files = readSet('fruits-360/Test')
test_size = int(len(files)/2)
test_set = files[:test_size]
valid_set = files[test_size:]

test_images, test_labels = zip(*test_set)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
