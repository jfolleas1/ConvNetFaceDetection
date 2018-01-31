import os, cv2, random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns


from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils
from keras.models import load_model


TRAIN_1_DIR = '../data/train/1/'
TRAIN_0_DIR = '../data/train/0/'
HARD_EXAMPLE_DIR = 'data_save_difficult_no_faces/'


ROWS = 36
COLS = 36
CHANNELS = 1


TRIAN_1_PATH = list(filter(lambda x: '.DS' not in x,[TRAIN_1_DIR+i for i in os.listdir(TRAIN_1_DIR)]))
TRIAN_0_PATH = list(filter(lambda x: '.DS' not in x,[TRAIN_0_DIR+i for i in os.listdir(TRAIN_0_DIR)]))


NB_TEST_BY_CLASS = 3000

NB_EPOCH = 5
BATCH_SIZE = 32

##
# Read the images at the indicate path and return a vectorized image
##
def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE) #
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)

##
# Takes a lists of images path and return a list of vectorized images
##
def prep_data(images):
    count = len(images)
    data = np.ndarray((count, ROWS, COLS), dtype=np.uint8)

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image
    
    return data

TRIAN_1_IMAGES = prep_data(TRIAN_1_PATH)
TRIAN_0_IMAGES = prep_data(TRIAN_0_PATH)

##
# shuffle the set of train images
##
def shuffle_and_get_new_train_set():
    random.shuffle(TRIAN_1_IMAGES)
    random.shuffle(TRIAN_0_IMAGES)
    test_images_1 = TRIAN_1_IMAGES[:NB_TEST_BY_CLASS]
    test_images_0 = TRIAN_0_IMAGES[:NB_TEST_BY_CLASS]
    train_1 = TRIAN_1_IMAGES[NB_TEST_BY_CLASS:]
    train_0 = TRIAN_0_IMAGES[NB_TEST_BY_CLASS:]
    return train_1, train_0, test_images_1, test_images_0

##
# reutrn the train set and the test set of images and coresponding labels 
##
def prepar_train_images():
    hard_example_paths = list(filter(lambda x: '.DS' not in x,[HARD_EXAMPLE_DIR+i for i in os.listdir(HARD_EXAMPLE_DIR)]))
    hard_example_set = prep_data(hard_example_paths)
    train_1, train_0, test_images_1, test_images_0 = shuffle_and_get_new_train_set()
    print("inside function : " + str(len(hard_example_set)))
    train_images = np.array(list(train_1[:(len(train_0)+len(hard_example_set))]) +
                            list(train_0) + list(hard_example_set))
    train_images.resize((len(train_images), 36, 36, 1))
    train_and_label = list(zip(train_images, ([1]*(len(train_images)//2)) + ([0]*(len(train_images)//2))))
    random.shuffle(train_and_label)
    train_images = list(map(lambda x: x[0], train_and_label))
    train_labels = list(map(lambda x: x[1], train_and_label))
    test_imagies = list(test_images_1) + list(test_images_0)
    test_imagies = np.array(test_imagies)
    test_imagies.resize((NB_TEST_BY_CLASS*2, 36, 36, 1))
    return np.array(train_images), np.array(train_labels), test_imagies

##
# Create the untrained model
##
def faceRecognition():
    
    model = Sequential()

    model.add(Conv2D(4, 5, strides=(1,1), border_mode='same',
                     input_shape=(36, 36, 1), data_format="channels_last", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(14, 3, strides=(1,1), border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(14, activation='relu'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=1e-4), metrics=['accuracy'])
    return model

##
# train the model on the train set with the hard examples and return it
##
def get_model(verbose_train=0):
    train_images, train_labels, test_imagies = prepar_train_images()
    print("LEN train : " + str(len(train_images)))


    model = faceRecognition()
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')
    model.fit(train_images, train_labels, batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH,
                validation_split=0.25, verbose=verbose_train, shuffle=True, callbacks=[early_stopping])
    return model

