import cv2
import numpy as np
from patchify import patchify
import utility


def predict_mask(path_to_image, dense_model, model, patch_size=224, window_stride=64):
    patchsize = patch_size
    stride = window_stride

    image1 = cv2.imread(path_to_image)
    imgpatches = patchify(image1, (patchsize, patchsize, 3), step=stride)
    mask1 = np.zeros((image1.shape[0], image1.shape[1]))
    mask2 = np.zeros((image1.shape[0], image1.shape[1]))

    patches1 = utility.convert_patches_to_list(imgpatches)
    patches1 = utility.preprocessing_patcheslist(patches1)
    features = dense_model.predict(patches1)
    features = np.reshape(features, (-1, 1, 1024))
    scoreslist = model.predict(features)
    scoreslist = np.reshape(scoreslist, (len(scoreslist)))

    for i in range(len(scoreslist)):
        col = i // int(imgpatches.shape[1])
        row = i % int(imgpatches.shape[1])

        x1 = (row * stride)
        y1 = (col * stride)
        x2 = x1 + patchsize
        y2 = y1 + patchsize

        mask1[y1:y2, x1:x2] = mask1[y1:y2, x1:x2] + scoreslist[i]
        mask2[y1:y2, x1:x2] = mask2[y1:y2, x1:x2] + 1

    mask1[np.isnan(mask1)] = 0
    mask2[mask2 == 0] = 1
    b = mask1 / mask2
    print("Mask generated of image " + str(0 + 1) + "/1")
    return b, path_to_image
