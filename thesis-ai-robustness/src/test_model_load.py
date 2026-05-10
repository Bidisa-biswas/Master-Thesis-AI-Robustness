import os
from tensorflow.keras.models import load_model

# Use the correct path (go up one level, then into the repo)
model_path = '../Stock-Price-Movement-Prediction/best_enhanced_model.keras'

# Check if file exists
if os.path.exists(model_path):
    model = load_model(model_path)
    print("✅ Model loaded successfully!")
    print(model.summary())
else:
    print(f"❌ File not found: {model_path}")
    print(f"Current working directory: {os.getcwd()}")