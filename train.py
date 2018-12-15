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

print(classesToIds)
print(idsToClasses)

def readSet(dir):
    files = []

    for i, className in enumerate(listdir(dir)):
        if className == '.DS_Store':
            continue
        ind = classesToIds[className]
        classDir = dir + '/' + className
        for filename in listdir(classDir):
            path = classDir + '/' + filename
            files.append((path, ind))
    return files


files = readSet('fruits-360/Training')

def preprocess_image(file):
    image = ndimage.imread(file)
    result_amplitude = np.zeros((100, 100, 6))
    for i in range(3):
        fft = np.fft.fft2(image[:, :, i])
        fshift = np.fft.fftshift(fft)
        result_amplitude[:, :, i] = np.abs(fshift) / 2445836.0
        result_amplitude[:, :, 3+i] = np.angle(fshift) / 3.14
    return result_amplitude

"""
max_amplitude = 0
max_phase = 0

for file in files:
    result_amplitude, result_phase = preprocess_image(file[0])
    max_amplitude = max(max_amplitude, np.max(result_amplitude))
    max_phase = max(max_phase, np.max(result_phase))

print(max_amplitude, max_phase)
"""

batch_size=100
def generate_arrays_from_files(files2):
    print(len(files2))
    batch_features = np.zeros((batch_size, 100, 100, 6))
    batch_labels = np.zeros((batch_size, 1))

    while 1:
        for i in range(batch_size):
            index = random.choice(files2)
            file, clas = index
            #print(file, clas)
            batch_features[i] = preprocess_image(file)
            batch_labels[i] = clas
        yield (batch_features, batch_labels)


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


model.fit_generator(generate_arrays_from_files(files), samples_per_epoch=10000, epochs=5)
model.save('my_model.h5')





