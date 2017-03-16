#!/usr/bin/env python3
"""This is test of keras library."""

import matplotlib.pyplot as plt
# import numpy as np
from keras.datasets import mnist
from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential
from keras.utils import np_utils


def main():
    """main function"""
    nb_classes = 10
    train_data, test_data = load_data()
    plot_example_input_data(train_data[0], train_data[1])
    train_data, test_data = process_data(train_data, test_data, nb_classes)
    trained_model = train_net(create_model(), train_data, test_data)
    evaluate_net(trained_model, test_data)


def plot_example_input_data(x_data, y_data, count=9):
    """Plot example input data."""
    for i in range(count):
        plt.subplot(3, 3, i + 1)
        plt.imshow(x_data[i], cmap='gray', interpolation='none')
        plt.title("Class {}".format(y_data[i]))


def load_data():
    """Load data from MNIST database."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print("x_train original shape", x_train.shape)
    print("y_train original shape", y_train.shape)
    print("x_test original shape", x_test.shape)
    print("y_test original shape", y_test.shape)

    return ((x_train, y_train), (x_test, y_test))


def process_data(train, test, nb_classes):
    """Process and trasform data to readable format in which it can be used to train neurala network."""
    (x_train, y_train) = train
    (x_test, y_test) = test
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    print("Training matrix shape", x_train.shape)
    print("Testing matrix shape", x_test.shape)
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    return ((x_train, y_train), (x_test, y_test))


def create_model():
    """Creates apropriate model for the input data."""
    model = Sequential()
    model.add(Dense(units=512, input_dim=(784, )))
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=10))
    model.add(Activation('softmax'))
    model.compile(
        loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model


def train_net(model, train_data, test_data):
    """Trains Neural Network with available data."""
    model.fit(
        train_data[0],
        train_data[1],
        batch_size=128,
        nb_epoch=4,
        show_accuracy=True,
        verbose=1,
        validation_data=test_data)
    return model


def evaluate_net(model, test_data):
    """Evaluates preformamce of the network on the test data."""
    score = model.evaluate(
        test_data[0], test_data[1], show_accuarcy=True, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    return score


# if __name__ == "__main__":
main()
