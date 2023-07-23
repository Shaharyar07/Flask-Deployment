from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
from flask_cors import CORS
from PIL import Image
import numpy as np

import io

app = Flask(__name__)
CORS(app)
# Load the model
model = load_model('./model.h5')
@app.route("/")
def home():
    return "<h1>Server is running</h1>"
 
@app.route('/predict', methods=['POST'])
def predict():
    # Load the image from the request
    image = Image.open(io.BytesIO(request.files['file'].read()))

    # Preprocess the image so it matches the input format of your model
    image = preprocess_image(image)

    # Make a prediction
    prediction = model.predict(image)


    # Convert the prediction to a JSON-compatible format
    prediction = prediction.tolist()
    if(prediction[0][0] < 0.5):
        return "Cancer"
    elif(prediction[0][0] > 0.5):
        return "Healthy"
    else:
        return "Unknown"


 

   
def preprocess_image(image):
    # Convert the PIL Image to a numpy array
    image = img_to_array(image)

    # Resize the image
    image = tf.image.resize(image, (224, 224))

    # Expand the dimensions to match the input shape of your model
    image = np.expand_dims(image, axis=0)

    return image

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
