import numpy as np
import warnings

from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.engine.topology import get_source_inputs

def identity_block(input_tensor, kernel_size, filters, stage, block, activation):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = Activation(activation)(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation(activation)(x)
    return x



def conv_block(input_tensor, kernel_size, filters, stage, block, activation, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = Activation(activation)(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = Activation(activation)(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation(activation)(x)
    return x

def resnetModel(img_input, activation, cut_layers):
    print("AAA", activation)
    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(1, 1), name='conv1')(x)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation(activation)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), activation=activation)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', activation=activation)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', activation=activation)

    if cut_layers > 1:
        x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', activation=activation)
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', activation=activation)
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', activation=activation)
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', activation=activation)

        if cut_layers > 2:
            x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', activation=activation)
            x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', activation=activation)
            x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', activation=activation)
            x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', activation=activation)
            x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', activation=activation)
            x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', activation=activation)

            if cut_layers > 3:
                x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', activation=activation)
                x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', activation=activation)
                x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', activation=activation)

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    x = Flatten()(x)
    x = Dense(84, activation='softmax', name='fc1000')(x)
    return x
    
