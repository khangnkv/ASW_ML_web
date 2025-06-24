#!/usr/bin/env python3
"""
Startup script for the ML Prediction System React frontend
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

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

def check_frontend_dependencies():
    """Check if frontend dependencies are installed"""
    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return False
    
    node_modules = frontend_dir / "node_modules"
    if not node_modules.exists():
        print("📦 Installing frontend dependencies...")
        os.chdir(frontend_dir)
        try:
            subprocess.run(["npm", "install"], check=True)
            print("✅ Frontend dependencies installed")
        except subprocess.CalledProcessError:
            print("❌ Failed to install frontend dependencies")
            return False
        finally:
            os.chdir("..")
    else:
        print("✅ Frontend dependencies already installed")
    
    return True

def start_frontend():
    """Start the React development server"""
    frontend_dir = Path("frontend")
    os.chdir(frontend_dir)
    
    print("🎯 Starting React development server...")
    print("📱 Frontend will be available at: http://localhost:3000")
    print("🌐 Backend should be running at: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        subprocess.run(["npm", "start"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Frontend server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error starting frontend server: {e}")
        sys.exit(1)
    finally:
        os.chdir("..")

def main():
    """Main startup function"""
    print("🚀 Starting ML Prediction System Frontend")
    print("=" * 50)
    
    # Check Node.js
    if not check_node_installed():
        sys.exit(1)
    
    # Check dependencies
    if not check_frontend_dependencies():
        sys.exit(1)
    
    # Start frontend
    start_frontend()

if __name__ == "__main__":
    main() 