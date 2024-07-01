import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model("BrainTumor10Epochs.h5")

image_path = "C:\\Users\\sp696\\OneDrive\\Desktop\\minor\\pred\\pred2.jpg"

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
image = Image.fromarray(image)
image = image.resize((64, 64))
image = np.array(image)
image = image / 255.0  # Normalize image

input_img = np.expand_dims(image, axis=0)

predictions = model.predict(input_img)
predicted_class = np.argmax(predictions, axis=1)

print("Predicted class:", predicted_class)
