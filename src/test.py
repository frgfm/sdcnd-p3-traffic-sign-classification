#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Traffic sign inference
'''

import argparse
from pathlib import Path
import glob
import cv2
import numpy as np
import pandas as pd

import json
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from utils import convert_to_grayscale, normalize


def main(args):

    # Load data
    data_folder = Path(args.folder)
    if not data_folder.is_dir():
        raise FileNotFoundError(f"unable to access {data_folder}")

    class_names = pd.read_csv(data_folder.joinpath('signnames.csv')).values[:, 1]

    model = load_model(args.model)

    # Load web images
    if not Path(args.imgfolder).is_dir():
        raise FileNotFoundError(f"unable to access {args.imgfolder}")
    files = glob.glob(args.imgfolder + '/*.png')
    X_web_images = np.empty((len(files), 32, 32, 3), dtype=np.uint8)
    y_web_images = np.empty(len(files), dtype=np.uint8)

    for idx, file in enumerate(files):
        X_web_images[idx, ...] = cv2.cvtColor(cv2.resize(cv2.imread(file), (32, 32)), cv2.COLOR_BGR2RGB)
        y_web_images[idx] = int(file.replace(args.imgfolder + '/', '').split('-')[0])

    # Check web images
    with open(data_folder.joinpath('normalization.json'), 'rb') as f:
        mean, std = json.load(f)
    X_web_images = normalize(convert_to_grayscale(X_web_images), mean, std)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.evaluate(x=X_web_images, y=y_web_images)

    # Print predictions
    out = model.predict(x=X_web_images)
    top_idxs = out.argsort(axis=1)[:, -5:][:, ::-1]
    for idx, img in enumerate(X_web_images):
        # Image + GT
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(img[..., 0], aspect='equal', cmap='gray')
        ax1.set_title(class_names[y_web_images[idx]])
        ax1.axis('off')
        # Top Prob
        ax2.barh(np.arange(5) + .5, out[idx, top_idxs[idx, :]])
        ax2.yaxis.tick_right()
        ax2.set_yticks(np.arange(5) + .5)
        ax2.set_yticklabels([class_names[idx] for idx in top_idxs[idx, :]])
        ax2.set_xlabel('Probability')
        # Plot setup
        plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Traffic sign classification inference',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("imgfolder", type=str, help="Path to image folder")
    parser.add_argument("--folder", type=str, default='./data', help="Images to test")
    parser.add_argument("--model", type=str, default='./data/model.h5',
                        help="Path to model checkpoint")

    args = parser.parse_args()
    main(args)
