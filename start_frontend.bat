@echo off
echo 🚀 Starting ML Prediction System Frontend
echo ================================================

cd frontend

echo 📦 Checking if node_modules exists...
if not exist "node_modules" (
    echo 📥 Installing dependencies...
    npm install
    if errorlevel 1 (
        echo ❌ Failed to install dependencies
        pause
        exit /b 1
    )
)

echo 🎯 Starting React development server...
echo 📱 Frontend will be available at: http://localhost:3000
echo 🌐 Backend should be running at: http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo ================================================

npm start

pause 