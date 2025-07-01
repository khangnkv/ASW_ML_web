#!/usr/bin/env python3
"""
Startup script for the ML Prediction System Flask backend
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import flask
        import pandas
        import numpy
        import joblib
        import openpyxl
        print("✓ All Python dependencies are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_models():
    """Check if model files exist"""
    model_dir = Path("model")
    if not model_dir.exists():
        print("✗ Model directory not found")
        return False
    
    model_files = list(model_dir.glob("project_*_model.pkl"))
    if not model_files:
        print("✗ No model files found in model/ directory")
        print("Expected format: project_{id}_model.pkl")
        return False
    
    print(f"✓ Found {len(model_files)} model files")
    return True

def create_upload_dir():
    """Create uploads directory if it doesn't exist"""
    backend_dir = Path(__file__).parent / 'backend'
    upload_dir = backend_dir / "uploads"
    upload_dir.mkdir(exist_ok=True)
    print("✓ Uploads directory ready")

def main():
    """Main startup function"""
    print("🚀 Starting ML Prediction System Backend")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check models
    if not check_models():
        print("⚠️  Warning: No models found. The system will work but predictions will fail.")
    
    # Create upload directory
    create_upload_dir()
    
    print("\n🎯 Starting Flask server...")
    print("📱 Backend will be available at: http://localhost:5000")
    print("🌐 Frontend should be available at: http://localhost:3000")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 50)
    
    # Start Flask app
    try:
        sys.path.insert(0, str(Path(__file__).parent / 'backend'))
        from app_workflow import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()