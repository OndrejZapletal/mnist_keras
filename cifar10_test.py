#!/usr/bin/env python3
"""This is test of keras library."""

import sys

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten,
                          MaxPooling2D)
from keras.models import Sequential
from keras.utils import np_utils


def main():
    """main function"""

    data = process_data(*load_data(), 10)
    model_protypes = []

    # optimizers = ['rmsprop', 'adam', 'nadam']
    optimizers = ['adam', 'nadam']

    model_protypes.append(create_model_1)
    model_protypes.append(create_model_2)
    model_protypes.append(create_model_3)
    model_protypes.append(create_model_4)
    model_protypes.append(create_model_5)

    find_optimal_model(model_protypes, optimizers, data, False)


def find_optimal_model(models, optimizers, data, plot):
    """ Function will test results of different optimizers.
    """
    for i, model in enumerate(models, start=1):
        for optimizer in optimizers:
            print('\nTesting model %s with optimizer %s' % (i, optimizer))
            trained_model = train_model(model(), data[0], optimizer)
            save_model_to_file(trained_model, '%s_%s' % (i, optimizer))
            evaluate_net(trained_model, data[1], plot)

def save_model_to_file(model, name):
    """Save model to json and weigths to h5py."""
    try:
        with open("model_%s_architecture.json" % name, 'w') as json_file:
            json_file.write(model.to_json())
        model.save_weights("model_%s_weights.h5" % name)
        print("model_%s saved" % name)
        return True
    except:
        print(sys.exc_info())
        return False


def plot_example_input_data(x_data, y_data, count=9):
    """Plot example input data."""
    for i in range(count):
        plt.subplot(3, 3, i + 1)
        plt.imshow(x_data[i], cmap='gray', interpolation='none')
        plt.title("Class {}".format(y_data[i]))


def load_data():
    """Load data from CIFAR10 database."""
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    return ((x_train, y_train), (x_test, y_test))


def process_data(train, test, nb_classes, max_val=255):
    """Process and trasform data to readable format in which it can be used to
    train neurala network.
    """
    (x_train, y_train) = train
    (x_test, y_test) = test
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= max_val
    x_test /= max_val
    # print("Training matrix shape", x_train.shape)
    # print("Testing matrix shape", x_test.shape)

    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    return ((x_train, y_train), (x_test, y_test))


def create_model_1(shape=(32, 32, 3), num_classes=10):
    """Creates apropriate model for the input data."""
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=shape))
    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model


def create_model_2(shape=(32, 32, 3), num_classes=10):
    """Creates apropriate model for the input data."""
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same',
                     input_shape=shape))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model


def create_model_3(shape=(32, 32, 3), num_classes=10):
    """Creates apropriate model for the input data."""
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same',
                     input_shape=shape))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model


def create_model_4(shape=(32, 32, 3), num_classes=10):
    """Creates apropriate model for the input data."""
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same',
                     input_shape=shape))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model


def create_model_5(shape=(32, 32, 3), num_classes=10):
    """Creates apropriate model for the input data."""
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=shape))
    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model


def train_model(model, train_data, optimizer):
    """Trains Neural Network with available data."""
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'])
    model.fit(
        train_data[0],
        train_data[1],
        batch_size=128,
        epochs=10,
        verbose=1,
        validation_split=0.1)
    return model


def evaluate_net(model, test_data, plot=False):
    """Evaluates preformamce of the network on the test data."""
    score = model.evaluate(test_data[0], test_data[1], verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    predicted_classes = model.predict(test_data[0])
    predicted_classes_indexes = [np.argmax(item) for item in predicted_classes]
    test_data_indexes = [np.argmax(item) for item in test_data[1]]

    correct_indexes = [
        i for i, _ in enumerate(test_data_indexes)
        if test_data_indexes[i] == predicted_classes_indexes[i]
    ]
    incorrect_indexes = [
        i for i, _ in enumerate(test_data_indexes)
        if test_data_indexes[i] != predicted_classes_indexes[i]
    ]

    print("Correctly guessed: %s" % len(correct_indexes))
    print("Incorrectly guessed: %s" % len(incorrect_indexes))

    if plot:
        plot_mistakes(incorrect_indexes, test_data, predicted_classes_indexes,
                      test_data_indexes)
    return score


def plot_mistakes(incorrect_indexes, test_data, predicted_classes_indexes,
                  test_data_indexes, number=10):

    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
               'horse', 'ship', 'truck')

    for i in range(number):
        plt.figure()
        plt.imshow(test_data[0][incorrect_indexes[i]])
        plt.title("Predicted {}, Class {}".format(
            classes[predicted_classes_indexes[incorrect_indexes[i]]],
            classes[test_data_indexes[incorrect_indexes[i]]]))


if __name__ == "__main__":
    main()
