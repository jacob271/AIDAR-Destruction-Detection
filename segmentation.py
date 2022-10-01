#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:05:48 2019

@author: hec
"""
import os
from optparse import OptionParser

import numpy as np

import Network
import crf
import results
import utility


def perform_segmentation(model_name, path_to_image, apply_crf):
    model_densenet = utility.load_denseNet()

    model_path = os.path.join("models", model_name)

    if os.path.exists(model_path):

        print("loading Trained model..")
        model = Network.model_attention()
        model.load_weights(model_path)
        mask, name = results.predict_mask(path_to_image=path_to_image, denseModel=model_densenet, model=model,
                                          patch_size=224, window_stride=64)
        mask2 = np.where(mask > 0.6, 255, 0)

        if apply_crf:

            print("Applying Crf...")
            crf_mask = crf.perform_crf(mask, path_to_image)
            return crf_mask
        else:
            return mask2
    else:
        print("Error: Model path not found")
        return


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--model_name", dest="model_name", help="Path to Model to calculate accuracy of test data.")
    parser.add_option("--test_path", dest="test_path", help="Path to test data.")
    parser.add_option("--apply_CRF", dest="apply_CRF", help="Mention if you want to apply CRF 'yes' or 'no' ")
    parser.add_option("--patch_stride", dest="patch_stride", type="int",
                      help="mention stride of window to extract patches and features.", default=64)
    (options, args) = parser.parse_args()

    perform_segmentation(options.model_name, options.test_path, options.apply_CRF == "yes")
