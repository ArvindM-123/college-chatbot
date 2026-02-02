import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")
OUTPUT_PATH = os.path.join(BASE_DIR, "model_weights.npz")

def export_weights():
    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    
    weights = []
    
    # Iterate through layers and extract weights for Dense layers
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            w, b = layer.get_weights()
            weights.append(w)
            weights.append(b)
            print(f"Extracted weights for layer: {layer.name} | Shape: {w.shape}")
    
    if len(weights) != 6: # 3 Dense layers * 2 (weights + bias)
        print("WARNING: Expected 6 weight arrays (3 layers), found", len(weights))
    
    np.savez_compressed(OUTPUT_PATH, w1=weights[0], b1=weights[1], 
                                     w2=weights[2], b2=weights[3], 
                                     w3=weights[4], b3=weights[5])
    
    print(f"✅ Successfully saved weights to {OUTPUT_PATH}")

if __name__ == "__main__":
    export_weights()
