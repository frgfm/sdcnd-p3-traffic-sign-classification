#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Model definition
'''

from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt


def lenet5(dropout_prob=0.2):
    """Implements a LeNet5 architecture with dropout

    Args:
        dropout_prob (float): dropout probability

    Returns:
        tensorflow.keras.models.Sequential: LeNet5 model
    """

    layers = [
        Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), padding='valid',
               activation='relu', data_format='channels_last', input_shape=(32, 32, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(16, (5, 5), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(120, activation='relu')
    ]

    if dropout_prob > 0:
        layers.append(Dropout(dropout_prob))
    layers.append(Dense(84, activation='relu'))
    if dropout_prob > 0:
        layers.append(Dropout(dropout_prob))
    layers.append(Dense(43, activation='softmax'))

    return Sequential(layers)


def show_feature_map(image_input, model, layer_name, activation_min=-1, activation_max=-1):
    """Display activation map of a specific layer

    Args:
        image_input (numpy.ndarray[1, H, W, 1]): input image
        model (tensorflow.keras.models.Sequential): image classification model
        layer_name (str): name of the feature layer to visualize
        activation_min (float): minimum value to display
        activation_max (float): maximum value to display
    """
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    activation = intermediate_layer_model.predict(image_input)
    featuremaps = activation.shape[3]
    plt.figure(1, figsize=(15, 15))
    for featuremap in range(featuremaps):
        plt.subplot(6, 8, featuremap + 1)  # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap))  # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0, :, :, featuremap], interpolation="nearest",
                       vmin=activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min != -1:
            plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", cmap="gray")
        plt.axis('off')
