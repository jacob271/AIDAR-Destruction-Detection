import cv2
import numpy as np
from PIL import Image
from fastapi.responses import StreamingResponse

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
    cv2.imwrite("image.jpg", image)

    mask = seg.perform_segmentation("Model1_AttentionNetwork_500.h5", "image.jpg", True)

    mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    image[mask == 255] = (36, 255, 12)

    cv2.imwrite("mask2.png", image)

    def iterfile():
        with open("mask2.png", mode="rb") as file_like:
            yield from file_like

    return StreamingResponse(iterfile(), media_type="image/png")
