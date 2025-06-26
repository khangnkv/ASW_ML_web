#!/usr/bin/env python3
"""
Test script to verify model loading
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.model.predictor import MLPredictor

def test_model_loading():
    """Test if models can be loaded successfully"""
    print("Testing model loading...")
    
    try:
        predictor = MLPredictor()
        print(f"✓ Successfully loaded {len(predictor.models)} models")
        
        # List some loaded models
        model_ids = list(predictor.models.keys())[:5]
        print(f"Sample loaded models: {model_ids}")
        
        return True
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("\n✅ All models loaded successfully!")
    else:
        print("\n❌ Model loading failed!")
        sys.exit(1) 