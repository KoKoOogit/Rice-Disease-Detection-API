import base64
import json
from fastapi import FastAPI
import tensorflow as tf
import cv2
import numpy as np
from pydantic import BaseModel
import uvicorn
import time
import os
from PIL import Image, ImageOps

app = FastAPI()
labels = ["BrownSpot", "Healthy", "Hispa", "LeafBlast"]
model = tf.keras.models.load_model("./RiceDiseaseDetection60.h5")

##check if folder exists, not create
if  not os.path.exists("./tmp"):
    os.makedirs("./tmp")

class ImageModel(BaseModel):
    image_url: str


def detect(file_path):

    data = np.ndarray(shape=(1, 200, 200, 3), dtype=np.float32)
# Replace this with the path to your image
    image = Image.open(file_path)
#resize the image to a 224x224 with the same strategy as in TM2:
#resizing the image to be at least 224x224 and then cropping from the center
    size = (200, 200)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

#turn the image into a numpy array
    image_array = np.asarray(image)
# Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
# Load the image into the array
    data[0] = normalized_image_array

# run the inference
    prediction = model.predict(data)
    result = np.argmax(prediction)


    return labels[result]


def write_file(save_path, data):
    with open(save_path, "wb") as f:
        f.write(base64.b64decode(data))
        f.close()
    return save_path    


@app.get('/')
def query_records():

    return ({'status': 'success'})
# create a base model

@app.post("/detect")
async def predict_image(image:ImageModel):
    plant_pic = image.image_url.split(",")[1]
    total_img = str(len(os.listdir("./tmp")))
    try:
        saved_img_path = write_file(f'./tmp/leaf{total_img}.jpg',plant_pic)
        predict_label = detect(saved_img_path)
    except Exception as e:
        print(e)
        predict_label = {
            "detail":str(e)
        }    

    returned_json =     {
        "img_label":predict_label
    }    

    return returned_json
