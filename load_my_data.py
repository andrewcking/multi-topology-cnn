
import cv2
import numpy as np

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

"""HELPER MODULE TO LOAD DATA"""

def load_my_data(img_rows, img_cols,batch_size, topology,train_data_dir,validation_data_dir):
    train_datagen = ImageDataGenerator(rescale=1./255)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_rows, img_cols),
            batch_size=batch_size,
            class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_rows, img_cols),
            batch_size=batch_size,
            class_mode='categorical')


    return train_generator, validation_generator
