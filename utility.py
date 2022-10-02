import numpy as np
import tensorflow as tf
from keras.models import Model


def load_denseNet():
    densenet = tf.keras.applications.densenet.DenseNet121(include_top=True, weights='imagenet', input_tensor=None,
                                                          input_shape=None, pooling=None, classes=1000)

    fclast = densenet.get_layer("avg_pool").output
    modeldense = Model(inputs=densenet.input, outputs=fclast)
    return modeldense


def convert_patches_to_list(patches):
    patcheslist = []
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            patcheslist.append(patches[i][j][0])

    return patcheslist


def preprocessing_patcheslist(patcheslist):
    # Converting into numpy and preprocessing
    patcheslist = np.array(patcheslist)
    patcheslist = patcheslist.astype('float16')
    patcheslist = patcheslist / 255.

    return patcheslist
