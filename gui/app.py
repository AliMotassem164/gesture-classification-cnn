import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --------------------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/vgg_model.h5")

model = load_model()
# --------------------------------------------------------------

with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

class_map = {
    "00": "Palm",
    "01": "L",
    "02": "Fist",
    "03": "Fist Moved",
    "04": "Thumb",
    "05": "Index",
    "06": "OK",
    "07": "Palm Moved",
    "08": "C",
    "09": "Down"
}


# --------------------------------------------------------------

IMG_SIZE = (224, 224)

def prepare_image(img):
    img = img.resize(IMG_SIZE)
    img = np.array(img)

    if img.shape[-1] == 4:
        img = img[..., :3]

    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# --------------------------------------------------------------

st.set_page_config(page_title="VGG Hand Gesture", layout="centered")

st.title("✋ Hand Gesture Recognition")
st.write("Upload an image and let VGG16 predict the gesture")

# --------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Hand Gesture Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

# --------------------------------------------------------------

if uploaded_file and st.button("Predict"):
    with st.spinner("Predicting..."):
        processed_image = prepare_image(image)
        prediction = model.predict(processed_image)

        class_index = np.argmax(prediction)
        confidence = np.max(prediction)

    raw_class = class_names[class_index]
    gesture_name = class_map.get(raw_class, raw_class)

    st.success(f"Gesture: {gesture_name}")

    st.info(f"Confidence: {confidence * 100:.2f}%")
