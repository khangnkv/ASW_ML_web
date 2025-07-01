@echo off
setlocal EnableDelayedExpansion

echo 🚀 Starting ML Prediction System
echo ================================================

:: Check if curl is available
where curl >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: curl is required but not found in PATH
    echo Please install curl and add it to your system PATH
    pause
    exit /b 1
)

:: Start backend first
echo 📡 Starting backend server...
start "ML Backend" cmd /c "start_backend.bat"

:: Wait for backend to be ready
echo ⏳ Waiting for backend to be ready...
:check_backend
timeout /t 2 /nobreak >nul
curl -s http://localhost:5000/health >nul 2>&1
if errorlevel 1 (
    echo 🔄 Backend not ready yet, waiting...
    goto check_backend
)

echo ✅ Backend is ready!
echo.

:: Start frontend
echo 📡 Starting frontend...
start "ML Frontend" cmd /c "start_frontend.bat"

echo ✨ System startup complete!
echo ================================================
echo 🌐 Backend URL: http://localhost:5000
echo 🌐 Frontend URL: http://localhost:3000
echo.
echo Press any key to exit this window. The system will continue running.
pause >nul
