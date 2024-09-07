import os
import tensorflow as tf
import numpy as np
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from pymongo import MongoClient
from datetime import datetime

app = Flask(__name__)

# Load the model, ensure the path is correct and no trailing spaces
model_path = 'potato.keras'
model = load_model(model_path)
print('Model loaded. Check http://127.0.0.1:5000/')

# Label dictionary, update according to your model
labels = {0: 'Healthy', 1: 'Potato___healthy',2:'Potato___leaf_spot',3:'Powdery',4:'Rust',5:'Tomato___healthy',6:'Tomato___leaf_spot'}
# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['plant_disease_db']
collection = db['predictions']

def getResult(image_path):
    # Ensure target size matches the input size of the model
    img = load_img(image_path, target_size=(150, 150))  # Adjusted to 150x150
    x = img_to_array(img)
    x = x.astype('float32') / 255.0
    x = np.expand_dims(x, axis=0)
    
    # Make predictions
    predictions = model.predict(x)[0]
    return predictions

def store_result_in_mongodb(image_filename, predicted_label):
    # Prepare the document to insert
    document = {
        'image_filename': image_filename,
        'predicted_label': predicted_label,
        'timestamp': datetime.utcnow()
    }
    
    # Insert the document into MongoDB
    collection.insert_one(document)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        predictions = getResult(file_path)
        predicted_label = labels[np.argmax(predictions)]
        
        # Store the result in MongoDB
        store_result_in_mongodb(f.filename, predicted_label)
        
        return str(predicted_label)
    return None

if __name__ == '__main__':
    app.run(debug=True)
