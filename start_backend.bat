@echo off
echo 🚀 Starting ML Prediction System Backend
echo ================================================

echo 📦 Checking Python dependencies...
python -c "import flask, pandas, numpy, joblib, openpyxl" 2>nul
if errorlevel 1 (
    echo 📥 Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Failed to install dependencies
        pause
        exit /b 1
    )
)

echo 🎯 Starting Flask server...
echo 📱 Backend will be available at: http://localhost:5000
echo 🌐 Frontend should be available at: http://localhost:3000
echo.
echo Press Ctrl+C to stop the server
echo ================================================

python start_backend.py

pause 