from tensorflow.keras.models import load_model
import tensorflow as tf

# Load the Keras model
model = load_model("BrainTumor10Epochs.h5")

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tf_lite_model = converter.convert()

# Save the TensorFlow Lite model
with open("model.tflite", "wb") as f:
    f.write(tf_lite_model)
