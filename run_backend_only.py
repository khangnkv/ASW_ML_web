#!/usr/bin/env python3
"""
Simple backend-only runner for ML Prediction System
Use this if you have npm/Node.js issues
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_dependencies():
    """Check if Python dependencies are installed"""
    print("🔍 Checking Python dependencies...")
    
    try:
        import flask
        import pandas
        import numpy
        import joblib
        print("✅ Python dependencies OK")
        return True
    except ImportError as e:
        print(f"❌ Missing Python dependency: {e}")
        print("Run: python setup.py")
        return False

def check_models():
    """Check if model files exist"""
    model_dir = Path("model")
    if not model_dir.exists():
        print("⚠️  Model directory not found")
        return False
    
    model_files = list(model_dir.glob("project_*_model.pkl"))
    if not model_files:
        print("⚠️  No model files found in model/ directory")
        print("   Expected format: project_{id}_model.pkl")
        return False
    
    print(f"✅ Found {len(model_files)} model files")
    return True

def create_directories():
    """Create necessary directories"""
    directories = ["uploads"]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def main():
    """Main function"""
    print("🚀 ML Prediction System - Backend Only")
    print("=" * 50)
    
    # Check dependencies
    if not check_python_dependencies():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Check models
    check_models()
    
    print("\n🎯 Starting Flask backend...")
    print("📱 Backend will be at: http://localhost:5000")
    print("🌐 You can test the API endpoints directly")
    print("\n📋 Available endpoints:")
    print("   - POST /api/upload - Upload file")
    print("   - POST /api/predict - Generate predictions")
    print("   - GET /api/export/<format>/<filename> - Export results")
    print("   - GET /api/models - Get available models")
    print("\n💡 Test with curl or Postman:")
    print("   curl -X POST -F 'file=@sample_data.csv' http://localhost:5000/api/upload")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 