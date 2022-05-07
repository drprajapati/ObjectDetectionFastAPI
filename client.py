import os
import cv2
import requests
import numpy as np
from IPython.display import Image, display
import io

base_url = 'http://127.0.0.1:8000'
endpoint = '/predict'
model = 'yolov3-tiny'
url_with_endpoint = base_url + endpoint
full_url = url_with_endpoint + "?model=" + model

dir_name = "images_predicted"
if not os.path.exists(dir_name):
    os.mkdir(dir_name)


def response_from_server(url, image_file, verbose=True):
    files = {'file': image_file}
    response = requests.post(url=url, files=files)
    status_code = response.status_code
    if verbose:
        msg = "Everything went well!" if status_code == 200 else "There was an error when handling the request."
        print(msg)
    return response


with open("images/apple.jpeg", "rb") as image_file:
    prediction = response_from_server(full_url, image_file)


def display_image_from_response(response):
    image_stream = io.BytesIO(response.content)
    image_stream.seek(0)
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    filename = "image_with_objects.jpeg"
    cv2.imwrite(f'images_predicted/{filename}', image)
    display(Image(f'images_predicted/{filename}'))


display_image_from_response(prediction)
