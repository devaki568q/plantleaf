import os
import numpy as np
from PIL import Image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# Load the model
model_path = 'all.keras'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
model = load_model(model_path)
print('Model loaded. Check http://127.0.0.1:5000/')

# Print model summary to check input shape
model.summary()

# Define labels
labels = {
    0: 'Tomato___Bacterial_spot',
    1: 'Tomato___Early_blight',
    2: 'Tomato___healthy',
    3: 'Tomato___Late_blight',
    4: 'Tomato___Leaf_Mold',
    5: 'Tomato___Septoria_leaf_spot',
    6: 'Tomato___Spider_mites',
    7: 'Tomato___Target_Spot',
    8: 'Tomato___Tomato_mosaic_virus',
    9: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
}

def get_result(image_path):
    try:
        # Adjust target_size to match model input
        img = load_img(image_path, target_size=(200, 200))  # Ensure target_size matches model input
        x = img_to_array(img)
        x = x.astype('float32') / 255.0
        x = np.expand_dims(x, axis=0)  # Add batch dimension
        predictions = model.predict(x)[0]
        return predictions
    except Exception as e:
        print(f"Error during image processing or prediction: {e}")
        return None

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Ensure the 'uploads' directory exists
    upload_folder = 'uploads'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    file_path = os.path.join(upload_folder, secure_filename(file.filename))
    file.save(file_path)

    # Get prediction
    try:
        predictions = get_result(file_path)
        if predictions is None:
            return "Error in processing the image", 500
        predicted_label = labels.get(np.argmax(predictions), "Unknown label")
        return str(predicted_label)
    except Exception as e:
        print(f"Error in prediction route: {e}")
        return f"Error in prediction: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
