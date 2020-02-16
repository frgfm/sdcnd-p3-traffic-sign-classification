#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Feature visualization script
'''

import argparse
from pathlib import Path
import cv2
import numpy as np
import json

import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from utils import convert_to_grayscale, normalize
from models import show_feature_map


def main(args):

    # Load data
    model = load_model(args.model)

    # Load web images
    if not Path(args.img).is_file():
        raise FileNotFoundError(f"unable to access {args.img}")
    X_web_images = np.empty((1, 32, 32, 3), dtype=np.uint8)

    X_web_images[0, ...] = cv2.cvtColor(cv2.resize(cv2.imread(args.img), (32, 32)), cv2.COLOR_BGR2RGB)

    # Check web images
    data_folder = Path(args.folder)
    if not data_folder.is_dir():
        raise FileNotFoundError(f"unable to access {data_folder}")
    with open(data_folder.joinpath('normalization.json'), 'rb') as f:
        mean, std = json.load(f)
    X_web_images = normalize(convert_to_grayscale(X_web_images), mean, std)

    show_feature_map(X_web_images, model, layer_name=args.layer)
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Traffic sign classification activation visualization',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("img", type=str, help="Path to image")
    parser.add_argument("--layer", type=str, default='conv2d',
                        help="Layer name (options: conv2d, max_pooling2d, conv2d_1, max_pooling2d_1)")
    parser.add_argument("--folder", type=str, default='./data', help="Images to test")
    parser.add_argument("--model", type=str, default='./data/model.h5',
                        help="Path to model checkpoint")

    args = parser.parse_args()
    main(args)
