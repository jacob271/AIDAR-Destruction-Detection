import os
from optparse import OptionParser
import model as mdl
import crf
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
    final_mask = mask1 / mask2
    print("Mask generated of image " + str(path_to_image))
    return final_mask, path_to_image


def perform_segmentation(path_to_image, apply_crf):
    model_densenet = utility.load_denseNet()
    model_path = os.path.join("models", "Model1_AttentionNetwork_500.h5")

    if not os.path.exists(model_path):
        print("Error: model not available..")

    print("loading Trained model..")
    model = mdl.model_attention()
    model.load_weights(model_path)
    mask, name = predict_mask(path_to_image=path_to_image, dense_model=model_densenet, model=model,
                                      patch_size=224, window_stride=64)
    mask2 = np.where(mask > 0.6, 255, 0)
    cv2.imwrite("tempmask.png", mask2)

    if apply_crf:
        print("Applying Crf...")
        crf_mask = crf.perform_crf(mask, path_to_image)
        return crf_mask
    else:

        return cv2.cvtColor(cv2.imread("tempmask.png"), cv2.COLOR_BGR2GRAY)


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--image_path", dest="image_path", help="Path to image.")
    parser.add_option("--apply_CRF", dest="apply_CRF", help="Mention if you want to apply CRF 'yes' or 'no' ")
    (options, args) = parser.parse_args()

    mask = perform_segmentation(options.test_path, options.apply_CRF == "yes")
    cv2.imwrite("mask.png", mask)
