@echo off
echo 🚀 ML Prediction System Setup
echo ================================================

echo 📦 Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Failed to install Python dependencies
    pause
    exit /b 1
)

echo 📦 Installing Node.js dependencies...
cd frontend
npm install
if errorlevel 1 (
    echo ❌ Failed to install Node.js dependencies
    pause
    exit /b 1
)
cd ..

echo ✅ Setup completed successfully!
echo.
echo 🎯 To start the system:
echo    1. Run start_backend.bat (or python start_backend.py)
echo    2. Run start_frontend.bat (or cd frontend && npm start)
echo.
echo 📱 Backend will be at: http://localhost:5000
echo 🌐 Frontend will be at: http://localhost:3000
echo.
echo 📁 Sample data file: sample_data.csv
echo.

pause 