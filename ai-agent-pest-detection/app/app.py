from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
from app.utils import get_class_indices

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = "saved_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Load class indices
class_indices = get_class_indices("DataSet/Train")
classes = list(class_indices.keys())

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty file'}), 400

    try:
        image = Image.open(file.stream).convert("RGB")
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array)
        predicted_class = classes[np.argmax(predictions[0])]

        return jsonify({'prediction': predicted_class}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
