import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import gradio as gr

# Paths
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/plant_disease_model.tflite"

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load class indices
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# Function to preprocess the image
def load_and_preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Prediction function
def predict_image_class(image):
    preprocessed_img = load_and_preprocess_image(image)
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], preprocessed_img)
    interpreter.invoke()
    
    # Get predictions
    predictions = interpreter.get_tensor(output_details[0]['index'])
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    
    return f"Prediction: {predicted_class_name}"

# Gradio Interface
interface = gr.Interface(
    fn=predict_image_class,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=gr.Textbox(label="Prediction"),
    title="🌱 Plant Disease Classifier (TFLite)",
    description="Upload a plant leaf image to classify its disease using a compressed TFLite model."
)

if __name__ == "__main__":
    interface.launch()