import os
import requests
import io
import streamlit as st

base_url = 'http://127.0.0.1:8000'
endpoint = '/predict'
model = 'yolov3-tiny'
url_with_endpoint = base_url + endpoint
full_url = url_with_endpoint + "?model=" + model

dir_name = "images_predicted"
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

st.title("Object Detection using YOLOv3")
upload = st.file_uploader("Upload an image")


def response_from_server(url, image_file, verbose=True):
    print(image_file)
    files = {'file': image_file}
    response = requests.post(url=url, files=files)
    status_code = response.status_code
    if verbose:
        msg = "Everything went well!" if status_code == 200 else "There was an error when handling the request."
        print(msg)
    return response


if upload is not None:
    col1, col2 = st.columns(2)
    bool_button = st.button("Predict")
    with col1:
        st.header("Original Image")
        image = upload.read()
        st.image(image)
        with open(os.path.join("images", upload.name), "wb") as f:
            f.write((upload).getbuffer())

    with col2:
        st.header("Detected Object")
        if bool_button:
            with open(f"images/{upload.name}", "rb") as image_file:
                prediction = response_from_server(full_url, image_file, verbose=False)
                image_stream = io.BytesIO(prediction.content)
                st.image(image_stream)
