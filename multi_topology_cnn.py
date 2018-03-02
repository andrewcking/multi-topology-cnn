import math

import keras
import numpy as np
from keras import applications
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

from load_my_data import load_my_data

####################
# PARAMETERS
####################
topology = 'VGG16'  # choices are 'inceptionv3' or 'resnet50' or 'VGG16' or 'VGG19'

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
batch_size = 32

epochs1 = 150
epochs2 = 50

# image parameters
img_width, img_height = 140, 140

train_generator, validation_generator = load_my_data(img_width, img_height, batch_size, topology, train_data_dir, validation_data_dir)

def save_bottlebeck_features():
    # build the VGG16 network
    if topology == 'VGG16':
        model = applications.VGG16(include_top=False, weights='imagenet', input_shape=(img_width, img_height, 3))
    elif topology == 'VGG19':
        model = applications.VGG19(include_top=False, weights='imagenet', input_shape=(img_width, img_height, 3))
    elif topology == 'inceptionv3':
        model = applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=(img_width, img_height, 3))
    elif topology == 'resnet50':
        model = applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(img_width, img_height, 3))
    elif topology == 'inceptionresnetv2':
        model = applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(img_width, img_height, 3))
    bottle_train_datagen = ImageDataGenerator(rescale=1./255)
    bottle_valid_datagen = ImageDataGenerator(rescale=1./255)

    generator = bottle_train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    nb_train_samples = len(generator.filenames)
    num_classes = len(generator.class_indices)
    predict_size_train = int(math.ceil(nb_train_samples / batch_size))
    bottleneck_features_train = model.predict_generator(generator, predict_size_train)
    np.save('bottleneck_features_train.npy', bottleneck_features_train)

    generator = bottle_valid_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    nb_validation_samples = len(generator.filenames)
    predict_size_validation = int(math.ceil(nb_validation_samples / batch_size))
    bottleneck_features_validation = model.predict_generator(generator, predict_size_validation)
    np.save('bottleneck_features_validation.npy', bottleneck_features_validation)


def train_top_model():
    num_classes = len(train_generator.class_indices)

    """PREPARE TOP BOTTLENECK DATA"""
    # load the bottleneck features saved earlier
    train_data = np.load('bottleneck_features_train.npy')
    # get the class lebels for the training data, in the original order
    train_labels = train_generator.classes
    # convert the training labels to categorical vectors
    train_labels = to_categorical(train_labels, num_classes=num_classes)
    # do the same for validation
    validation_data = np.load('bottleneck_features_validation.npy')
    validation_labels = validation_generator.classes
    validation_labels = to_categorical(validation_labels, num_classes=num_classes)

    """CREATE TOP"""
    if topology == 'VGG16'or topology == 'VGG19':
        model = Sequential()
        model.add(Flatten(input_shape=train_data.shape[1:]))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.65))
        model.add(Dense(num_classes, activation='sigmoid'))
    elif topology == 'inceptionv3':
        model = Sequential()
        model.add(GlobalAveragePooling2D(input_shape=train_data.shape[1:]))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.65))
        model.add(Dense(num_classes, activation='softmax'))
    elif topology == 'resnet50':
        model = Sequential()
        model.add(Flatten(input_shape=train_data.shape[1:]))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.65))
        model.add(Dense(num_classes, activation='softmax'))
    elif topology == 'inceptionresnetv2':
        model = Sequential()
        model.add(GlobalAveragePooling2D(input_shape=train_data.shape[1:]))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.65))
        model.add(Dense(num_classes, activation='softmax'))

    """COMPILE TRAIN SAVE"""
    sgd1 = SGD(lr=1e-3, decay=5e-4)
    model.compile(optimizer=sgd1, loss='categorical_crossentropy', metrics=['accuracy'])


    history = model.fit(train_data, train_labels,
                        epochs=epochs1,
                        batch_size=batch_size,
                        validation_data=(validation_data, validation_labels))

    model.save_weights(top_model_weights_path)


def train_whole_model():
    nb_train_samples = len(train_generator.filenames)
    nb_validation_samples = len(validation_generator.filenames)

    num_classes = len(train_generator.class_indices)

    top_model = Sequential()
    if topology == 'VGG16' or topology == 'VGG19':
        if topology == 'VGG16':
            base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
        elif topology == 'VGG19':
            base_model = applications.VGG19(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
        top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.65))
        top_model.add(Dense(num_classes, activation='softmax'))
    elif topology == 'inceptionv3':
        base_model = applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=(img_width, img_height, 3))
        top_model.add(GlobalAveragePooling2D(input_shape=base_model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.65))
        top_model.add(Dense(num_classes, activation='softmax'))
    elif topology == 'resnet50':
        base_model = applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(img_width, img_height, 3))
        top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.65))
        top_model.add(Dense(num_classes, activation='softmax'))
    elif topology == 'inceptionresnetv2':
        base_model = applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(img_width, img_height, 3))
        top_model.add(GlobalAveragePooling2D(input_shape=base_model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.65))
        top_model.add(Dense(num_classes, activation='softmax'))

    top_model.load_weights(top_model_weights_path)

    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))


    for layer in model.layers:
        layer.trainable = True

    sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.5, nesterov=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,  # floor div
        epochs=epochs2,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)
    model.save_weights('all_pretrained_weights.h5')
    model.save('all_pretrained_entire_model.h5')

save_bottlebeck_features()
train_top_model()
train_whole_model()
