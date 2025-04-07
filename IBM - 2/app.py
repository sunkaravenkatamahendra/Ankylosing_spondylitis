import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image

# Load your trained model
model = load_model(r"V:\Projects\Projects\IBM - 2\mobilenetv2_balanced_model.h5")

# Class names (must be in same order as training)
class_names = ['Normal', 'Unhealthy', 'Unhealthy']

# Title
st.title("🩻 Ankylosing Spondylitis")
st.write("Upload an X-ray image to detect possible diseases. And you should be in this format jpg , jpeg , png ")

# File uploader
uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = prediction[0][class_index]

    st.markdown("---")
    st.subheader("🔍 Prediction Result")
    st.write(f"**Predicted Class:** `{class_names[class_index]}`")
    st.write(f"**Confidence Score:** `{confidence:.2f}`")

    if class_names[class_index] == "Normal":
        st.success("✅ The image appears to be normal.")
    else:
        st.error(f"⚠️ The image shows signs of `{class_names[class_index]}`.")
