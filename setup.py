#!/usr/bin/env python3
"""
Setup script for ML Prediction System
Replaces the .bat file with a cross-platform Python solution
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_python_dependencies():
    """Install Python dependencies"""
    print("🐍 Installing Python dependencies...")
    
    # Try to upgrade pip first
    run_command("python -m pip install --upgrade pip", "Upgrading pip")
    
    # Install dependencies with more flexible versions
    success = run_command("pip install -r requirements.txt", "Installing Python packages")
    
    if not success:
        print("⚠️  Trying alternative installation method...")
        # Try installing packages one by one
        packages = [
            "flask",
            "flask-cors", 
            "pandas",
            "numpy",
            "openpyxl",
            "joblib",
            "werkzeug"
        ]
        
        for package in packages:
            if not run_command(f"pip install {package}", f"Installing {package}"):
                print(f"❌ Failed to install {package}")
                return False
    
    return True

def check_node_installed():
    """Check if Node.js is installed"""
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Node.js {result.stdout.strip()} detected")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ Node.js not found. Please install Node.js from https://nodejs.org/")
    return False

def install_node_dependencies():
    """Install Node.js dependencies"""
    if not check_node_installed():
        return False
    
    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return False
    
    # Change to frontend directory
    os.chdir(frontend_dir)
    
    # Check if node_modules exists
    if not Path("node_modules").exists():
        print("📦 Installing Node.js dependencies...")
        success = run_command("npm install", "Installing npm packages")
        if not success:
            return False
    else:
        print("✅ Node.js dependencies already installed")
    
    # Go back to root directory
    os.chdir("..")
    return True

def create_directories():
    """Create necessary directories"""
    directories = ["uploads", "frontend/build"]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

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

def main():
    """Main setup function"""
    print("🚀 ML Prediction System Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install Python dependencies
    if not install_python_dependencies():
        print("❌ Failed to install Python dependencies")
        sys.exit(1)
    
    # Install Node.js dependencies
    if not install_node_dependencies():
        print("❌ Failed to install Node.js dependencies")
        sys.exit(1)
    
    # Check models
    check_models()
    
    print("\n✅ Setup completed successfully!")
    print("\n🎯 To start the system:")
    print("   1. Run: python start_backend.py")
    print("   2. Run: python start_frontend.py")
    print("\n📱 Backend will be at: http://localhost:5000")
    print("🌐 Frontend will be at: http://localhost:3000")
    print("\n📁 Sample data file: sample_data.csv")
    print("\n💡 Alternative commands:")
    print("   - Backend: python app.py")
    print("   - Frontend: cd frontend && npm start")

if __name__ == "__main__":
    main() 