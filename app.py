import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# === Load your model ===
model = load_model("emotion_model.keras")  # Update if needed

# === Define class labels in order used during training ===
class_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']  # Update to your labels

# === Set image size (should match training input) ===
IMG_SIZE = (48, 48)

# === Prediction function ===
def predict_emotion(img):
    try:
        # Convert to grayscale if trained on grayscale images
        img = img.convert("L")

        # Resize
        img = img.resize(IMG_SIZE)

        # Convert to array, normalize
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 48, 48, 1)
        img_array = np.expand_dims(img_array, axis=-1) if img_array.shape[-1] != 1 else img_array

        # Predict
        predictions = model.predict(img_array)[0]
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions)

        return f"{predicted_class} ({confidence * 100:.2f}%)"

    except Exception as e:
        return f"Error: {str(e)}"

# === Gradio Interface ===
interface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Image(type="pil", label="Upload a face image"),
    outputs=gr.Textbox(label="Predicted Emotion"),
    title="Emotion Detection",
    description="Upload a face image to get the predicted emotion using a trained CNN model.",
)

# === Run the App ===
if __name__ == "__main__":
    interface.launch(debug=True)