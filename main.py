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

    cv2.imwrite("mask.png", mask)

    def iterfile():
        with open("mask.png", mode="rb") as file_like:
            yield from file_like

    return StreamingResponse(iterfile(), media_type="image/png")
