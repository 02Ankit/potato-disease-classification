import tensorflow as tf
from google.cloud import storage
from PIL import Image
import numpy as np 


BUCKET_NAME = 'bucket-potato-disease-classification'
class_names = ["Early Blight", "Late Blight", "Healthy"]
model = None

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # model upload ko lane ka tarika hai storage client ko
    storage_client = storage.Client() 
    # Storage client me se bucket ko get karenge
    bucket = storage_client.get_bucket(bucket_name)
    # or use buckete me se blob(binary live object) lenge jo ki basicaly humara model hai
    blob = bucket.blob(source_blob_name)

    # or use blob ko google bucket ke storge me se destinationfile name ka matlab jaha jis cloud ke server pe wo hai use server pe koun se location me store karna hai.
    blob.download_to_filename(destination_file_name)

    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")


def predict(request):
    global model #first time agar koi model load karega to directly load hoga 
    # but second time koi load karega to wo global model save ho chuka hoga
    # to if condition false hoga  
    if model is None: # agar model none hai 
        #actualy google ka jo bucket hai use humare google k server pe download ya save karna hai 
        # to hum download_blob()ka use karenge
        download_blob(
            BUCKET_NAME,
            "models/potatoes.h5", # source path bucket se lenge
            "/tmp/potatoes.h5", #destination tmp directory me store karadenge
        )
        # model ko load karenge
        model = tf.keras.models.load_model("/tmp/potatoes.h5")

    image = request.files["file"] # file is a keyy postman me dekh chuke hai 

    image = np.array(
        Image.open(image).convert("RGB").resize((256, 256)) # image resizing
    )

    image = image/255 # normalize the image in 0 to 1 range

    img_array = tf.expand_dims(image, 0)
    predictions = model.predict(img_array)

    print("Predictions:",predictions)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)

    return {"class": predicted_class, "confidence": confidence}



