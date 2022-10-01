import os
from optparse import OptionParser
import numpy as np
import Network
import crf
import results
import utility
import cv2


def perform_segmentation(path_to_image, apply_crf):
    model_densenet = utility.load_denseNet()
    model_path = os.path.join("models", "Model1_AttentionNetwork_500.h5")

    if not os.path.exists(model_path):
        print("Error: model not available..")

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


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--image_path", dest="image_path", help="Path to image.")
    parser.add_option("--apply_CRF", dest="apply_CRF", help="Mention if you want to apply CRF 'yes' or 'no' ")
    (options, args) = parser.parse_args()

    mask = perform_segmentation(options.test_path, options.apply_CRF == "yes")
    cv2.imwrite("mask.png", mask)
