import cv2
import numpy as np
from PIL import Image

import segmentation as seg
from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/seg_image")
def get_image(file: UploadFile = File(...)):
    image = np.array(Image.open(file.file))
    cv2.imwrite(image)
    return seg.perform_segmentation("Model1_AttentionNetwork_500.h5", ".", "masks", True, "crf_masks")
