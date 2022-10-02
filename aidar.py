import cv2
import numpy as np
from PIL import Image
from fastapi.responses import StreamingResponse
from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
import segmentation as seg

app = FastAPI()


@app.post("/segment_image")
def segment_image(file: UploadFile = File(...)):
    image = np.array(Image.open(file.file))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("image.jpg", image)
    mask = seg.perform_segmentation("image.jpg", False)
    mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    image[mask == 255] = (111, 114, 255)
    cv2.imwrite("mask.png", image)

    def iterfile():
        with open("mask.png", mode="rb") as file_like:
            yield from file_like

    return StreamingResponse(iterfile(), media_type="image/png")
