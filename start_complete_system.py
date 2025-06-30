#!/usr/bin/env python3
"""
Complete System Startup Script for ML Prediction System
Ensures backend starts first and stays stable before starting frontend
"""

import subprocess
import time
import requests
import sys
import os
from pathlib import Path

def check_backend_health(max_retries=10, delay=2):
    """Check if backend is healthy and ready to accept requests"""
    print("Checking backend health...")
    
    for attempt in range(max_retries):
        try:
            response = requests.get('http://localhost:5000/api/health', timeout=5)
            if response.status_code == 200:
                print("✅ Backend is healthy and ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        print(f"⏳ Backend not ready yet (attempt {attempt + 1}/{max_retries})")
        time.sleep(delay)
    
    print("❌ Backend failed to start or become healthy")
    return False

def start_backend():
    """Start the backend server"""
    print("=" * 50)
    print("  Starting Backend Server")
    print("=" * 50)
    
    backend_dir = Path(__file__).parent / 'backend'
    cmd = [sys.executable, 'app_workflow.py']
    
    try:
        # Start backend process
        process = subprocess.Popen(
            cmd,
            cwd=backend_dir,
            creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
        )
        print(f"Backend process started with PID: {process.pid}")
        
        # Wait a moment for startup
        time.sleep(5)
        
        # Check if backend is healthy
        if check_backend_health():
            return process
        else:
            print("Backend startup failed!")
            process.terminate()
            return None
            
    except Exception as e:
        print(f"Error starting backend: {e}")
        return None

def start_frontend():
    """Start the frontend development server"""
    print("\n" + "=" * 50)
    print("  Starting Frontend Development Server")
    print("=" * 50)
    
    frontend_dir = Path(__file__).parent / 'frontend'
    cmd = ['npm', 'start']
    
    try:
        # Start frontend process
        process = subprocess.Popen(
            cmd,
            cwd=frontend_dir,
            creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
        )
        print(f"Frontend process started with PID: {process.pid}")
        return process
        
    except Exception as e:
        print(f"Error starting frontend: {e}")
        return None

def main():
    """Main startup sequence"""
    print("🚀 ML Prediction System - Complete Startup")
    print("=" * 50)
    
    # Step 1: Start backend
    backend_process = start_backend()
    if not backend_process:
        print("❌ Failed to start backend. Exiting.")
        return 1
    
    # Step 2: Start frontend
    frontend_process = start_frontend()
    if not frontend_process:
        print("❌ Failed to start frontend. Terminating backend.")
        backend_process.terminate()
        return 1
    
    # Step 3: Show status
    print("\n" + "=" * 50)
    print("  System Status")
    print("=" * 50)
    print("✅ Backend: Running on http://localhost:5000")
    print("✅ Frontend: Running on http://localhost:3000")
    print("\nBoth servers are now running.")
    print("Press Ctrl+C to stop all servers.")
    print("=" * 50)
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nShutting down servers...")
        if backend_process:
            backend_process.terminate()
        if frontend_process:
            frontend_process.terminate()
        print("All servers stopped.")
        return 0

if __name__ == '__main__':
    sys.exit(main())
