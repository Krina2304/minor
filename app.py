import os
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = load_model("BrainTumor10Epochs.h5")
print('Model loaded. Check http://127.0.0.1:5000/')


def get_className(classNo):
    if classNo == 0:
        return "No Brain Tumor"
    elif classNo == 1:
        return "Yes Brain Tumor"


def getResult(img):
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = Image.fromarray(image)
    image = image.resize((64, 64))
    image = np.array(image)
    image = image / 255.0  # Normalize image
    input_img = np.expand_dims(image, axis=0)
    result = model.predict(input_img)
    class_idx = np.argmax(result, axis=1)[0]
    className = get_className(class_idx)
    return className


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        file_path = os.path.join('uploads', secure_filename(f.filename))
        f.save(file_path)
        result = getResult(file_path)
        return result
    return "Prediction failed"


if __name__ == '__main__':
    app.run(debug=True)
