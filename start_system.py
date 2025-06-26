#!/usr/bin/env python3
"""
Startup script for the ML Prediction System
Starts both backend and frontend with proper error handling
"""

import subprocess
import time
import sys
import os
import requests
from pathlib import Path

def check_port_available(port):
    """Check if a port is available"""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    return result != 0

def wait_for_backend(url, max_attempts=30):
    """Wait for backend to be ready"""
    print(f"Waiting for backend at {url}...")
    for i in range(max_attempts):
        try:
            response = requests.get(f"{url}/api/health", timeout=2)
            if response.status_code == 200:
                print("✅ Backend is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
        if (i + 1) % 5 == 0:
            print(f"Still waiting... ({i + 1}/{max_attempts})")
    return False

def main():
    print("🚀 Starting ML Prediction System...")
    
    # Check if we're in the right directory
    if not Path("backend").exists() or not Path("frontend").exists():
        print("❌ Error: Please run this script from the project root directory")
        sys.exit(1)
    
    # Check if backend port is available
    if not check_port_available(5000):
        print("❌ Error: Port 5000 is already in use. Please stop any existing backend process.")
        sys.exit(1)
    
    # Check if frontend port is available
    if not check_port_available(3000):
        print("❌ Error: Port 3000 is already in use. Please stop any existing frontend process.")
        sys.exit(1)
    
    # Start backend
    print("🔧 Starting backend...")
    backend_process = None
    try:
        backend_process = subprocess.Popen(
            [sys.executable, "backend/app_workflow.py"],
            cwd=os.getcwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for backend to start
        if not wait_for_backend("http://localhost:5000"):
            print("❌ Backend failed to start within expected time")
            if backend_process:
                backend_process.terminate()
            sys.exit(1)
        
        # Start frontend
        print("🎨 Starting frontend...")
        frontend_process = subprocess.Popen(
            ["npm", "start"],
            cwd="frontend",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print("\n" + "="*60)
        print("🎉 ML Prediction System is starting up!")
        print("="*60)
        print("📊 Backend: http://localhost:5000")
        print("🌐 Frontend: http://localhost:3000")
        print("="*60)
        print("Press Ctrl+C to stop both services")
        print("="*60 + "\n")
        
        # Wait for processes
        try:
            backend_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
            if backend_process:
                backend_process.terminate()
            if frontend_process:
                frontend_process.terminate()
            print("✅ Services stopped")
            
    except Exception as e:
        print(f"❌ Error starting services: {e}")
        if backend_process:
            backend_process.terminate()
        sys.exit(1)

if __name__ == "__main__":
    main() 