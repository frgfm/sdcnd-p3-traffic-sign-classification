#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Image utilities function
'''


import numpy as np


def convert_to_grayscale(X):
    """Convert RGB images to grayscale

    Args:
        X (numpy.ndarray[N, H, W, 3]): RGB image

    Returns:
        numpy.ndarray[N, H, W, 1]: grayscale unit image
    """

    # Tensor placeholder
    processed_X = np.empty(X.shape[:-1] + (1,), dtype=np.float32)

    # Convert to grayscale
    processed_X[..., 0] = 0.299 * X[:, :, :, 0] + 0.587 * X[:, :, :, 1] + 0.114 * X[:, :, :, 2]

    # Unit values
    processed_X /= 255.

    return processed_X


def normalize(X, mean, std):
    """Normalize image with a given mean and standard deviation

    Args:
        X (numpy.ndarray[N, H, W, 1]): input tensor
        mean (float): mean to use for normalization
        std (float): standard deviation to use for normalization

    Returns:
        numpy.ndarray[N, H, W, 1]: normalized tensor
    """

    X -= mean
    X /= std

    return X
