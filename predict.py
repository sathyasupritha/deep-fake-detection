import os
import cv2
import joblib
import numpy as np

# Load the trained model
model_path = "deepfake_detector.pkl"
model = joblib.load(model_path)

# Image preprocessing function (same as used during training)
def preprocess_image(img_path, img_size=(64, 64)):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"❌ Could not read image: {img_path}")
    
    img = cv2.resize(img, img_size)
    img = img.flatten().astype(np.uint8)  # Convert to uint8 for consistency
    return img.reshape(1, -1)  # Reshape for model input

# Function to predict if an image is real or fake
def predict_image(img_path):
    try:
        img_data = preprocess_image(img_path)
        prediction = model.predict_proba(img_data)[0]  # Get probability
        fake_prob = prediction[1]  # Probability of being fake
        real_prob = prediction[0]  # Probability of being real
        
        if fake_prob > real_prob:
            print(f"🛑 Prediction: FAKE ({fake_prob * 100:.2f}% confidence)")
        else:
            print(f"✅ Prediction: REAL ({real_prob * 100:.2f}% confidence)")
    except Exception as e:
        print(f"Error: {e}")

# Test with a sample image
sample_image = "S:/Documents/6th sem/PR/Celeb-DF Preprocessed/test/fake/id0_id1_0000_frame180_face4.jpg"  # Change to your image path
predict_image(sample_image)
