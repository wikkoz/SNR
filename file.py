from os import listdir
from os.path import isfile, join
from scipy import fftpack, ndimage
import matplotlib.pyplot as plt
import numpy as np
import sys

np.set_printoptions(threshold=np.nan)
def readTrainingSet():
    paths = []
    training = 'fruits-360/Training'
    for className in listdir(training):
        classDir = training + '/' + className
        for filename in listdir(classDir):
            path = classDir + '/' + filename
            paths.append((path, className))
    return paths

def getImages(paths):
    images = []
    for (path, className) in paths:
        images.append((transformImage(ndimage.imread(path), className)))
    return images

def transformImage(images):
    result = np.zeros((100, 100, 6))
    for i in range(3):
        fft = np.fft.fft2(image[:, :, i])
        fshift = np.fft.fftshift(fft)
        result[:, :, 2 * i] = fshift.real
        result[:, :, 2 * i + 1] = fshift.imag
    return result

print (readTrainingSet())
