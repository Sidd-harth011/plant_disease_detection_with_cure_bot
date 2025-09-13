import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import gradio as gr
import requests
from huggingface_hub import hf_hub_download

# -----------------------------
# Load API key from Hugging Face Secrets
# -----------------------------
GROK_KEY = os.getenv("GROK_API_KEY")
GROK_URL = "https://api.groq.com/openai/v1/chat/completions"

# -----------------------------
# Load TFLite model from Hugging Face Hub
# -----------------------------
model_path = hf_hub_download(
    repo_id="sidd-harth011/checkingPDRMod",  # ✅ your repo
    filename="plant_disease_model.tflite"
)

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -----------------------------
# Load class indices (local file in repo)
# -----------------------------
class_indices = json.load(open("class_indices.json"))

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
# OpenAI Chatbot (single-turn, no history)
# -----------------------------
def grok_chatbot(user_message):
    payload = {
        "model": "openai/gpt-oss-20b",  # ✅ lightweight, works in Spaces
        "messages": [{"role": "user", "content": user_message}],
        "temperature": 0.7,
        "max_tokens": 500
    }

    headers = {
        "Authorization": f"Bearer {GROK_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(GROK_URL, headers=headers, json=payload)

    if response.status_code == 200:
        bot_message = response.json()["choices"][0]["message"]["content"]
    else:
        print("Error:", response.status_code, response.text)
        bot_message = "⚠️ Sorry, I couldn't process that. Try again!"

    return bot_message

# -----------------------------
# Gradio Interface
# -----------------------------
with gr.Blocks(title="🌱 Plant Disease Classifier & AI Chatbot (OpenAI)") as demo:

    gr.Markdown("## 🌱 Plant Disease Classifier with AI Assistant (OpenAI)")

    with gr.Row():
        # Left: Plant classifier
        with gr.Column(scale=1):
            gr.Markdown("### Upload Image")
            image_input = gr.Image(type="pil", label="Upload a Plant Leaf Image")
            predict_button = gr.Button("Classify")
            prediction_output = gr.Textbox(label="Prediction")

            predict_button.click(fn=predict_image_class, inputs=image_input, outputs=prediction_output)

        # Right: AI Chatbot
        with gr.Column(scale=1):
            gr.Markdown("### 🤖 AI Chatbot")
            msg = gr.Textbox(label="Type your message")
            response_box = gr.Textbox(label="Bot Response", lines=5)
            send_btn = gr.Button("Send")

            send_btn.click(grok_chatbot, inputs=msg, outputs=response_box)

if __name__ == "__main__":
    demo.launch()
