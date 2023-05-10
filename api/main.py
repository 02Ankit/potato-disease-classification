from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from io import BytesIO # convert bytes 
from PIL import Image   #read PIL(Pillow) module to use read Images in python

from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf 

print(tf.reduce_sum(tf.random.normal([1000, 1000])))
app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("./save_models/2") # here we upload trained Model 
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get(path="/ping")
async def ping():
    return "Hello , I am alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data))) # it will read thus bytes as in  pillow image and to convert this pillow image into numpy array
    return image

@app.post(path="/predict")
async def predict(file: UploadFile = File(...)): 
    #file: UploadFile = File(...) here file: is type hint, UploadFile is DataType 
    # =File(...) this is default value
   
    # bytes = await file.read() here file read as a Bytes so we need read file as a image we can use numpy 
    image = read_file_as_image(await file.read()) 
    print("OnlyImage:", image)
    image_batch = np.expand_dims(image, 0) # its take multi dimension array please check numpy official site you understand the concept 
    print("Image_batch",image_batch)
    predictions = MODEL.predict(image_batch)# it canot take only one image its take batch of images
    print(predictions)
    print(predictions[0])
    #np.argmax(predictions[0]) it is like ---> [0.3456, 0.99865, 0.566678] here it will take max value of an array 
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]

    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)