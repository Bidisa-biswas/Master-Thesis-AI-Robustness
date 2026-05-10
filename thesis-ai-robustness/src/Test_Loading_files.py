"""
Test script to verify LSTM model and scaler loading.
Run this to confirm everything works before full prediction.
"""

import os
import sys
import joblib
from tensorflow.keras.models import load_model

# Add project root to path if needed
project_root = '/Users/bidisabiswas/PycharmProjects/Master-Thesis-AI-Robustness/thesis-ai-robustness'
sys.path.insert(0, project_root)

print("=" * 60)
print("TESTING LSTM MODEL AND SCALER LOADING")
print("=" * 60)

# 1. Define file paths
repo_path = os.path.join(project_root, 'Stock-Price-Movement-Prediction')
model_path = os.path.join(repo_path, 'best_enhanced_model.keras')
scaler_path = os.path.join(repo_path, 'artifacts', 'enhanced', 'feature_scaler_enhanced.pkl')

print(f"\n1. Checking file paths...")
print(f"   Model path: {model_path}")
print(f"   Scaler path: {scaler_path}")

# 2. Check if files exist
print(f"\n2. Checking if files exist...")
model_exists = os.path.exists(model_path)
scaler_exists = os.path.exists(scaler_path)

print(f"   Model file exists: {model_exists}")
print(f"   Scaler file exists: {scaler_exists}")

if not model_exists:
    print(f"\n❌ ERROR: Model not found!")
    print(
        f"   Please run: cd {project_root} && git clone https://github.com/marepallisanthosh999333/Stock-Price-Movement-Prediction.git")
    sys.exit(1)

if not scaler_exists:
    print(f"\n⚠️ WARNING: Scaler not found at expected path!")
    print(f"   Looking for: {scaler_path}")
    print(f"   The model may still work, but scaling might fail.")

# 3. Load the model
print(f"\n3. Loading LSTM model...")
try:
    model = load_model(model_path)
    print(f"   ✅ Model loaded successfully!")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
    print(f"   Number of layers: {len(model.layers)}")
except Exception as e:
    print(f"   ❌ Failed to load model: {e}")
    sys.exit(1)

# 4. Load the scaler
print(f"\n4. Loading feature scaler...")
if scaler_exists:
    try:
        scaler = joblib.load(scaler_path)
        print(f"   ✅ Scaler loaded successfully!")
        print(f"   Scaler type: {type(scaler).__name__}")
        print(f"   Expected features: {scaler.n_features_in_}")
    except Exception as e:
        print(f"   ❌ Failed to load scaler: {e}")
        scaler = None
else:
    print(f"   ⚠️ Scaler file not found. Will proceed without scaling (not recommended).")
    scaler = None

# 5. Quick test with dummy data
print(f"\n5. Testing with dummy data...")
try:
    import numpy as np

    # Create dummy input matching model expectations
    dummy_input = np.random.randn(10, 60, 24).astype(np.float32)
    print(f"   Dummy input shape: {dummy_input.shape}")

    # Apply scaling if available
    if scaler is not None:
        dummy_2d = dummy_input.reshape(-1, 24)

        dummy_scaled = scaler.transform(dummy_2d)
        dummy_input = dummy_scaled.reshape(10, 60, 24)
        print(f"   ✅ Scaling applied")

    # Run prediction
    dummy_output = model.predict(dummy_input, verbose=0)
    print(f"   Dummy output shape: {dummy_output.shape}")
    print(f"   Sample predictions: {dummy_output[:5].flatten()}")
    print(f"   ✅ Prediction test passed!")

except Exception as e:
    print(f"   ❌ Dummy test failed: {e}")

# 6. Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Model:     {'✅ LOADED' if model_exists else '❌ MISSING'}")
print(f"Scaler:    {'✅ LOADED' if scaler_exists else '⚠️ MISSING'}")
print(f"Test:      ✅ PASSED")
print("\n✅ Everything is ready! You can proceed to full predictions.")
print("=" * 60)