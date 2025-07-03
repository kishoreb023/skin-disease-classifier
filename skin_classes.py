# Class mapping
class_name_map = {
    'akiec': "Actinic Keratoses / Intraepithelial Carcinoma",
    'bcc': "Basal Cell Carcinoma",
    'bkl': "Benign Keratosis-like Lesions",
    'df': "Dermatofibroma",
    'mel': "Melanoma (Skin Cancer)",
    'nv': "Melanocytic Nevi (Common Mole)",
    'vasc': "Vascular Lesions"
}

class_codes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# After predicting
pred = model.predict(img_array)  # shape: (1, 7)
predicted_index = np.argmax(pred)
predicted_code = class_codes[predicted_index]
predicted_label = class_name_map[predicted_code]
confidence = float(np.max(pred)) * 100

# Display result
st.success("✅ Prediction Result")
st.markdown(f"**{predicted_label}**")
st.info(f"🧠 Confidence: **{confidence:.2f}%**")
