import numpy as np
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from enum import Enum
from fastapi.responses import StreamingResponse
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox

app = FastAPI()


class Model(str, Enum):
    yolo_tiny: str = "yolov3-tiny"
    yolo_v3: str = "yolov3"


@app.get("/")
def home():
    return "Head Over to http://localhost:8000/docs"


@app.post("/predict")
def predict(model: Model, file: UploadFile = File(...)):
    filename = file.filename
    file_extension = filename.split(".")[-1] in ("jpeg", "png", "jpg")
    if not file_extension:
        raise HTTPException(status_code=415, detail="Unsupported file provided")

    image_stream = io.BytesIO(file.file.read())
    image_stream.seek(0)
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    cv2.imshow("Image", image)
    bbox, label, confidence = cv.detect_common_objects(image, model=model)
    output = draw_bbox(img=image, bbox=bbox, labels=label, confidence=confidence)
    cv2.imwrite(filename=f"images_uploaded/{filename}", img=output)
    output_image = open(f"images_uploaded/{filename}", mode="rb")
    return StreamingResponse(content=output_image, media_type='image/jpeg')
