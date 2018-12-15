from os import listdir
from os.path import isfile, join
from scipy import fftpack, ndimage
import matplotlib.pyplot as plt
import numpy as np


def readTrainingSet():
    training = 'fruits-360/Test'
    images = {}
    for className in listdir(training):
        classImages = []
        classDir = training + '/' + className
        for filename in listdir(classDir):
            path = classDir + '/' + filename
            classImages.append(ndimage.imread(path))
        images[className] = classImages
    print (images)

readTrainingSet()
