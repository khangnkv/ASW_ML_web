#!/usr/bin/env python3
"""
Complete system runner for ML Prediction System
Runs both backend and frontend in separate processes
"""

import os
import sys
import time
import subprocess
import threading
import signal
from pathlib import Path

class SystemRunner:
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.running = True
        
        # Handle graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\n🛑 Shutting down system...")
        self.running = False
        self.stop_processes()
        sys.exit(0)
    
    def find_npm(self):
        """Find npm executable on Windows"""
        # Common npm locations on Windows
        possible_paths = [
            "npm",
            "npm.cmd",
            r"C:\Program Files\nodejs\npm.cmd",
            r"C:\Program Files (x86)\nodejs\npm.cmd",
            os.path.expanduser(r"~\AppData\Roaming\npm\npm.cmd"),
        ]
        
        for path in possible_paths:
            try:
                result = subprocess.run([path, "--version"], capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"✅ Found npm: {path}")
                    return path
            except FileNotFoundError:
                continue
        
        return None
    
    def check_dependencies(self):
        """Check if all dependencies are installed"""
        print("🔍 Checking dependencies...")
        
        # Check Python dependencies
        try:
            import flask
            import pandas
            import numpy
            import joblib
            print("✅ Python dependencies OK")
        except ImportError as e:
            print(f"❌ Missing Python dependency: {e}")
            print("Run: python setup.py")
            return False
        
        # Check Node.js
        try:
            result = subprocess.run(["node", "--version"], capture_output=True, text=True)
            if result.returncode != 0:
                raise FileNotFoundError
            print(f"✅ Node.js {result.stdout.strip()} OK")
        except FileNotFoundError:
            print("❌ Node.js not found. Please install from https://nodejs.org/")
            return False
        
        # Check npm
        npm_path = self.find_npm()
        if not npm_path:
            print("❌ npm not found. Please ensure Node.js is properly installed.")
            print("Try reinstalling Node.js from https://nodejs.org/")
            return False
        
        # Check frontend dependencies
        frontend_dir = Path("frontend")
        if not (frontend_dir / "node_modules").exists():
            print("📦 Installing frontend dependencies...")
            os.chdir(frontend_dir)
            try:
                subprocess.run([npm_path, "install"], check=True)
                print("✅ Frontend dependencies installed")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install frontend dependencies: {e}")
                return False
            finally:
                os.chdir("..")
        else:
            print("✅ Frontend dependencies already installed")
        
        return True
    
    def start_backend(self):
        """Start the Flask backend (runs app_workflow.py)"""
        print("🚀 Starting Flask backend...")
        try:
            backend_path = Path("backend") / "app_workflow.py"
            if not backend_path.exists():
                # Try current directory (for Docker/local flexibility)
                backend_path = Path("app_workflow.py")
            if not backend_path.exists():
                print(f"❌ Could not find app_workflow.py at {backend_path}")
                return False
            self.backend_process = subprocess.Popen(
                [sys.executable, str(backend_path)],
                stdout=sys.stdout,
                stderr=sys.stderr,
                text=True
            )
            print(f"✅ Backend started (PID: {self.backend_process.pid})")
        except Exception as e:
            print(f"❌ Failed to start backend: {e}")
            return False
        return True
    
    def start_frontend(self):
        """Start the React frontend (npm start in frontend directory)"""
        print("🚀 Starting React frontend...")
        try:
            npm_path = self.find_npm()
            if not npm_path:
                print("❌ npm not found")
                return False
            frontend_dir = Path("frontend")
            if not frontend_dir.exists():
                print("❌ frontend directory not found")
                return False
            self.frontend_process = subprocess.Popen(
                [npm_path, "start"],
                cwd=str(frontend_dir),
                stdout=sys.stdout,
                stderr=sys.stderr,
                text=True
            )
            print(f"✅ Frontend started (PID: {self.frontend_process.pid})")
        except Exception as e:
            print(f"❌ Failed to start frontend: {e}")
            return False
        return True
    
    def wait_for_backend(self):
        """Wait for backend to be ready"""
        print("⏳ Waiting for backend to be ready...")
        max_attempts = 30
        for attempt in range(max_attempts):
            try:
                import requests
                response = requests.get("http://localhost:5000/api/models", timeout=1)
                if response.status_code == 200:
                    print("✅ Backend is ready!")
                    return True
            except:
                pass
            time.sleep(1)
        
        print("❌ Backend failed to start within 30 seconds")
        return False
    
    def monitor_processes(self):
        """Monitor running processes"""
        while self.running:
            # Check backend
            if self.backend_process and self.backend_process.poll() is not None:
                print("❌ Backend process stopped unexpectedly")
                self.running = False
                break
            
            # Check frontend
            if self.frontend_process and self.frontend_process.poll() is not None:
                print("❌ Frontend process stopped unexpectedly")
                self.running = False
                break
            
            time.sleep(2)
    
    def stop_processes(self):
        """Stop all running processes"""
        print("🛑 Stopping processes...")
        
        if self.backend_process:
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
            print("✅ Backend stopped")
        
        if self.frontend_process:
            self.frontend_process.terminate()
            try:
                self.frontend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.frontend_process.kill()
            print("✅ Frontend stopped")
    
    def run(self):
        """Main run method"""
        print("🚀 ML Prediction System")
        print("=" * 50)
        
        # Check dependencies
        if not self.check_dependencies():
            sys.exit(1)
        
        # Start backend
        if not self.start_backend():
            sys.exit(1)
        
        # Wait for backend to be ready
        if not self.wait_for_backend():
            self.stop_processes()
            sys.exit(1)
        
        # Start frontend
        if not self.start_frontend():
            self.stop_processes()
            sys.exit(1)
        
        print("\n🎉 System is running!")
        print("📱 Frontend: http://localhost:3000")
        print("🌐 Backend:  http://localhost:5000")
        print("\nPress Ctrl+C to stop the system")
        print("=" * 50)
        
        # Monitor processes
        try:
            self.monitor_processes()
        except KeyboardInterrupt:
            print("\n👋 Shutting down...")
        finally:
            self.stop_processes()

def main():
    """Main function"""
    runner = SystemRunner()
    runner.run()

if __name__ == "__main__":
    main() 