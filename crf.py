#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 14:25:17 2019

@author: hec
"""

import cv2
import imageio
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, unary_from_softmax
from skimage.io import imsave


def perform_crf(mask, image_name):
    img = cv2.imread(image_name)
    imsave("temp.png", mask)
    anno_rgb = imageio.imread("temp.png").astype(np.uint32)
    min_val = np.min(anno_rgb.ravel())
    max_val = np.max(anno_rgb.ravel())
    if (max_val - min_val) == 0:
        out = (anno_rgb.astype('float') - min_val) / 1
    else:
        out = (anno_rgb.astype('float') - min_val) / (max_val - min_val)
    labels = np.zeros((2, img.shape[0], img.shape[1]))
    labels[1, :, :] = out
    labels[0, :, :] = 1 - out

    colors = [0, 255]
    colorize = np.empty((len(colors), 1), np.uint8)
    colorize[:, 0] = colors

    n_labels = 2

    crf = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)

    U = unary_from_softmax(labels)
    crf.setUnaryEnergy(U)
    feats = create_pairwise_bilateral(sdims=(100, 100), schan=(13, 13, 13),
                                      img=img, chdim=2)
    crf.addPairwiseEnergy(feats, compat=10,
                          kernel=dcrf.FULL_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = crf.inference(20)
    MAP = np.argmax(Q, axis=0)
    MAP = colorize[MAP]
    return MAP.reshape(anno_rgb.shape)
