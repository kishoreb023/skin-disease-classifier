import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Set page config
st.set_page_config(page_title="Skin Disease Classifier", layout="centered")

st.title("🧪 Skin Disease Classifier")
st.caption("Upload a skin lesion image to automatically predict the disease.")

# Load trained model
model = tf.keras.models.load_model("best_skin_model.h5")

# Skin disease class codes and labels
class_name_map = {
    'akiec': "Actinic Keratoses / Intraepithelial Carcinoma",
    'bcc': "Basal Cell Carcinoma",
    'bkl': "Benign Keratosis-like Lesions",
    'df': "Dermatofibroma",
    'mel': "Melanoma (Skin Cancer)",
    'nv': "Melanocytic Nevi (Common Mole)",
    'vasc': "Vascular Lesions"
}
class_codes = list(class_name_map.keys())

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image", 
    type=["jpg", "jpeg", "png"],
    help="Upload a JPG or PNG image of a skin lesion.",
    label_visibility="visible"
)

# Display prediction only if image uploaded
if uploaded_file is not None:
    col1, col2 = st.columns(2)

    # Display uploaded image on the left
    with col1:
        st.image(uploaded_file, caption="Uploaded Image", width=300)

    # Predict and display on the right
    with col2:
        # Preprocess image
        image = Image.open(uploaded_file).resize((224, 224))
        img_array = np.expand_dims(np.array(image) / 255.0, axis=0)

        # Predict
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions)
        predicted_code = class_codes[predicted_index]
        predicted_label = class_name_map[predicted_code]
        confidence = float(np.max(predictions)) * 100

        # Display results
        st.success("✅ Prediction Result")
        st.markdown(f"**{predicted_label}**")
        st.info(f"🧠 Confidence: **{confidence:.2f}%**")

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
