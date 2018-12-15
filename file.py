from os import listdir
from os.path import isfile, join
from scipy import fftpack, ndimage
import matplotlib.pyplot as plt
import numpy as np


def readTrainingSet():
    training = 'fruits-360/Test'
    images = []
    for className in listdir(training):
        classDir = training + '/' + className
        for filename in listdir(classDir):
            path = classDir + '/' + filename
            images.append((transformImage(ndimage.imread(path)), className))

def transformImage(image):
    result = np.zeros((100, 100, 6))
    for i in range(3):
        fft = np.fft.fft2(image[:, :, i])
        fshift = np.fft.fftshift(fft)
        result[:, :, 2 * i] = fft.real
        result[:, :, 2 * i + 1] = fft.imag
    return result


readTrainingSet()
