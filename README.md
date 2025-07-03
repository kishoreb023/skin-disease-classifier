# Skin Disease Classification Web App 🩺

This Streamlit app uses a pre-trained CNN model (MobileNetV2) to predict common skin diseases from dermatoscopic images using the HAM10000 dataset.

## How to Run
1. Install dependencies:
   pip install -r requirements.txt

2. Start the app:
   streamlit run app.py

## Model Info
- Trained on: HAM10000 dataset
- Classes: akiec, bcc, bkl, df, mel, nv, vasc
- Architecture: MobileNetV2 + GlobalAveragePooling + Dense