from os import listdir
from scipy import fftpack, ndimage
import numpy as np
import pickle
import keras
import tensorflow as tf
import os.path
import random
import resnet
from argparse import ArgumentParser
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
from keras import losses
from skimage.transform import rescale, resize, downscale_local_mean

def prepareIds():
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

    return classesToIds, idsToClasses


def readDataSet(dir, classesToIds):
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


def random_rotation(image_array):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-90, 90)
    return sk.transform.rotate(image_array, random_degree, cval=1)


def random_noise(image_array):
    # add random noise to the image
    return sk.util.random_noise(image_array)


def horizontal_flip(image_array):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]


def augment(image):
    from random import randint
    funcs = [random_rotation, random_noise, horizontal_flip]
    for func in funcs:
        x = randint(0, 2)
        if x == 0:
            image = func(image)
    return image


def preprocess_image(file, size):
    x = ndimage.imread(file)
    if size != 100:
        return resize(x, (size, size))
    return x



def one_hot_encode(val):
    x = np.zeros(84)
    x[val] = 1
    return x

def train(args, classesToIds):
    def generate_arrays_from_files(batch_size, files2):
        print(len(files2))
        batch_features = np.zeros((batch_size, args.descriptor_size, args.descriptor_size, 3))

        if args.cost_function == 'sparse_categorical_crossentropy':
            batch_labels = np.zeros((batch_size, 1))
        else:
            batch_labels = np.zeros((batch_size, 84))

        while 1:
            for i in range(batch_size):
                index = random.choice(files2)
                file, clas = index
                # print(file, clas)
                batch_features[i] = preprocess_image(file, args.descriptor_size)
                if args.augmentation:
                    batch_features[i] = augment(batch_features[i])
                if args.cost_function == 'sparse_categorical_crossentropy':
                    batch_labels[i] = clas
                else:
                    batch_labels[i] = one_hot_encode(clas)
            yield (batch_features, batch_labels)

    files = readDataSet('fruits-360/Training', classesToIds)
    batch_size = args.batch_size
    samples = np.ceil(len(files) / batch_size)

    img_input = keras.layers.Input((args.descriptor_size, args.descriptor_size, 3))
    model = keras.Model(inputs=img_input, outputs=resnet.resnetModel(img_input, activation=args.activation_function, cut_layers=args.convolution_layers_cut))

    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=args.learning_rate),
                  loss=args.cost_function,
                  metrics=['accuracy'])

    model.summary()
    model.fit_generator(generate_arrays_from_files(batch_size, files), samples_per_epoch=samples, epochs=args.epochs)
    model.save('my_model.h5')
    return model

def test(args, classesToIds, model = None):
    def generate_arrays_from_files():
        print(len(test_set))
        i = 0
        while i < len(files):
            batch_features = np.zeros((batch_size, args.descriptor_size, args.descriptor_size, 3))
            if args.cost_function == 'sparse_categorical_crossentropy':
                batch_labels = np.zeros((batch_size, 1))
            else:
                batch_labels = np.zeros((batch_size, 84))
            j = 0
            while j < batch_size and i < len(files):
                file, clas = files[i]
                batch_features[j] = preprocess_image(file, args.descriptor_size)
                if args.cost_function == 'sparse_categorical_crossentropy':
                    batch_labels[j] = clas
                else:
                    batch_labels[j] = one_hot_encode(clas)
                i += 1
                j += 1
            yield (batch_features, batch_labels)

    if model is None:
        model = keras.models.load_model('my_model.h5')

    files = readDataSet('fruits-360/Test', classesToIds)
    test_size = int(len(files) / 2)
    test_set = files[:test_size]
    valid_set = files[test_size:]

    test_images, test_labels = zip(*test_set)

    batch_size = 100
    samples = int(len(files) / batch_size)
    test_loss, test_acc = model.evaluate_generator(generate_arrays_from_files(), steps=samples)
    print('Test accuracy:', test_acc)


def calc_model(args):
    classesToIds, idsToClasses = prepareIds()
    model = None
    if not args.only_test:
        model = train(args, classesToIds)
    test(args, classesToIds, model)

    pass

def str2bool(v):
    import argparse
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", dest="model", default="resNet", help="One of: resNet, SVM")
    parser.add_argument("-desc_size", "--descriptor_size", dest="descriptor_size", type=int, default=100, help="Descriptor size")
    parser.add_argument("-conv_cut", "--convolution_layers_cut", dest="convolution_layers_cut", default=4, type=int, help="Number from 1 to 4")
    parser.add_argument("-cost_fun", "--cost_function", dest="cost_function", default="mean_squared_error", help="One of: sparse_categorical_crossentropy, mean_squared_error, mean_absolute_error, binary_crossentropy, cosine_proximity etc")
    parser.add_argument("-activ_fun", "--activation_function", dest="activation_function", default="relu",
                        help="One of: relu, sigmoid, elu, softplus")
    parser.add_argument("-aug", "--augmentation", dest="augmentation", default='false', type=str2bool, help="If the data should be augmented")
    parser.add_argument("-batch_size", "--batch_size", dest="batch_size", default=20,
                        help="Batch size")
    parser.add_argument("-only_test", "--only_test", dest="only_test", default=False, help="Test only")
    parser.add_argument("-e", "--epochs", dest="epochs", default=3, type=int, help="Number of epochs")
    parser.add_argument("-lr", "--learning_rate", dest="learning_rate", default=0.01, type=float, help="Learning rate")

    args = parser.parse_args()

    calc_model(args)

if __name__ == '__main__':
    main()






