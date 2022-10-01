# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 17:25:54 2019

@author: usmal
"""
import glob
import os.path

import PIL
import cv2
import numpy as np

train_destructed_path = []
train_data_folder = os.path.join("Data", "train", "destructed", "*.jpg")
train_destructed_path.extend(glob.glob(train_data_folder))

if train_destructed_path is None:
    print("Error: Path not found to train data folder..")
else:
    print("Successfully read train data folder path..!")
    print("Total images: ", len(train_destructed_path))

# Apply augmentation Rotate images and save in train directory
for i in range(len(train_destructed_path)):
    imageName = train_destructed_path[i].split("/")[-1].split('.')[0]
    print(imageName)
    temp = cv2.imread(train_destructed_path[i])
    img1 = PIL.Image.fromarray(temp)
    writePath = os.path.join("Data", "train", "destructed")
    cv2.imwrite(writePath + imageName + "_1.jpg", np.array(img1.rotate(180)))
    cv2.imwrite(writePath + imageName + "_2.jpg", np.array(img1.rotate(270)))
    cv2.imwrite(writePath + imageName + "_3.jpg", np.array(img1.rotate(350)))
#
