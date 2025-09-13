import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import gradio as gr
from huggingface_hub import hf_hub_download

# -----------------------------
# Download TFLite model from Hugging Face model repo
# -----------------------------
model_path = hf_hub_download(
    repo_id="sidd-harth011/checkingPDRMod",  # your model repo
    filename="plant_disease_model.tflite"
)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -----------------------------
# Load class indices locally from project repo
# -----------------------------
working_dir = os.path.dirname(os.path.abspath(__file__))
class_indices_path = os.path.join(working_dir, "class_indices.json")
class_indices = json.load(open(class_indices_path))

# -----------------------------
# Preprocessing function
# -----------------------------
def load_and_preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# -----------------------------
# Prediction function
# -----------------------------
def predict_image_class(image):
    preprocessed_img = load_and_preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], preprocessed_img)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return f"Prediction: {predicted_class_name}"

# -----------------------------
# Gradio Interface
# -----------------------------
interface = gr.Interface(
    fn=predict_image_class,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=gr.Textbox(label="Prediction"),
    title="🌱 Plant Disease Classifier (TFLite)",
    description="Upload a plant leaf image to classify its disease using a compressed TFLite model hosted on Hugging Face."
)

if __name__ == "__main__":
    interface.launch()
