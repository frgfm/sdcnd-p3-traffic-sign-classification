#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Traffic sign classifcation training
'''

import argparse
from pathlib import Path
import pickle
import numpy as np
from utils import convert_to_grayscale, normalize
from models import lenet5
import json
import matplotlib.pyplot as plt


def main(args):

    # Load data
    data_folder = Path(args.folder)
    if not data_folder.is_dir():
        raise FileNotFoundError(f"unable to access {data_folder}")

    with open(data_folder.joinpath('train.p'), mode='rb') as f:
        train_set = pickle.load(f)
    with open(data_folder.joinpath('valid.p'), mode='rb') as f:
        valid_set = pickle.load(f)
    with open(data_folder.joinpath('test.p'), mode='rb') as f:
        test_set = pickle.load(f)

    X_train, y_train = train_set['features'], train_set['labels']
    X_valid, y_valid = valid_set['features'], valid_set['labels']
    X_test, y_test = test_set['features'], test_set['labels']

    # Dataset statistics
    sign_classes, class_indices, class_counts = np.unique(y_train, return_index=True, return_counts=True)

    print("Number of training examples:", X_train.shape[0])
    print("Number of validation examples:", X_valid.shape[0])
    print("Number of testing examples:", X_test.shape[0])
    print('Image data shape:', X_train.shape[1:])
    print('Number of classes:', class_counts.shape[0])

    # Class distribution
    plt.bar(np.arange(class_counts.shape[0]), class_counts, align='center')
    plt.xlabel('Class')
    plt.ylabel('Number of training examples')
    plt.xlim([-1, class_counts.shape[0]])
    plt.show()

    # Grayscale & unit values
    X_train = convert_to_grayscale(X_train)
    X_valid = convert_to_grayscale(X_valid)
    X_test = convert_to_grayscale(X_test)

    # Normalize
    mean, std = X_train.mean(), X_train.std()
    with open(data_folder.joinpath('normalization.json'), 'w') as f:
        json.dump((float(mean), float(std)), f)
    X_train = normalize(X_train, mean, std)
    X_valid = normalize(X_valid, mean, std)
    X_test = normalize(X_test, mean, std)

    # Model
    model = lenet5(dropout_prob=0.2)
    print(model.summary())

    # specify optimizer, loss function and metric
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # training batch_size=128, epochs=10
    _ = model.fit(X_train, y_train, batch_size=args.batch_size, epochs=args.epochs,
                  validation_data=(X_valid, y_valid))

    model.evaluate(x=X_test, y=y_test)
    model.save(data_folder.joinpath('model.h5'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Traffic sign classification training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("epochs", type=int, help="Number of epochs to train")
    parser.add_argument("--folder", type=str, default='./data', help="Path to data folder")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")

    args = parser.parse_args()
    main(args)
