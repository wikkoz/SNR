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
from sklearn.svm import SVC  



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
    return ndimage.imread(file)
    # result_amplitude = np.zeros((100, 100, 6))
    # for i in range(3):
    #     fft = np.fft.fft2(image[:, :, i])
    #     fshift = np.fft.fftshift(fft)
    #     result_amplitude[:, :, i] = np.abs(fshift) / 2445836.0
    #     result_amplitude[:, :, 3+i] = np.angle(fshift) / 3.14
    # return result_amplitude

"""
max_amplitude = 0
max_phase = 0

for file in files:
    result_amplitude, result_phase = preprocess_image(file[0])
    max_amplitude = max(max_amplitude, np.max(result_amplitude))
    max_phase = max(max_phase, np.max(result_phase))

print(max_amplitude, max_phase)
"""

batch_size=20
samples = np.ceil(len(files) / batch_size)

resnet_model = load_model('my_model.h5')
resnet_model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
svm_model = keras.Model(resnet_model.inputs, resnet_model.layers[-2].output)
svm_model.summary()
def generate_arrays_from_files(files2):
    print(len(files2))
    batch_features = np.zeros((batch_size, 100, 100, 3))
    batch_labels = np.zeros((batch_size))

    while 1:
        for i in range(batch_size):
            index = random.choice(files2)
            file, clas = index
            #print(file, clas)
            batch_features[i] = preprocess_image(file)
            batch_labels[i] = clas
        yield (svm_model.predict(batch_features), batch_labels)    

svclassifier = SVC(kernel='poly', degree=8, gamma='scale')
i = 0
for (batch_features, batch_labels) in generate_arrays_from_files(files):
    svclassifier.fit(batch_features, batch_labels)
    print(i)
    i += 1