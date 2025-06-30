@echo off
echo ================================================
echo   ML Prediction System - Complete Startup
echo ================================================

REM Kill any existing processes on ports 3000 and 5000
echo Cleaning up existing processes...
taskkill /F /IM node.exe /T 2>nul || echo No Node processes to kill
taskkill /F /IM python.exe /T 2>nul || echo No Python processes to kill

REM Wait a moment for cleanup
timeout /t 2 /nobreak >nul

echo.
echo ================================================
echo   Step 1: Starting Backend Server
echo ================================================

REM Start backend in a new window and keep it running
start "Backend Server" cmd /k "cd /d %~dp0backend && python app_workflow.py"

echo Backend starting... Waiting for it to be ready...
timeout /t 10 /nobreak >nul

REM Test backend health
echo Testing backend health...
curl -s http://localhost:5000/api/health >nul
if %errorlevel% neq 0 (
    echo WARNING: Backend may not be ready yet
    echo Waiting additional 5 seconds...
    timeout /t 5 /nobreak >nul
)

echo.
echo ================================================
echo   Step 2: Starting Frontend Development Server
echo ================================================

REM Start frontend in a new window
start "Frontend Server" cmd /k "cd /d %~dp0frontend && npm start"

echo.
echo ================================================
echo   System Status
echo ================================================
echo Backend: Running on http://localhost:5000
echo Frontend: Starting on http://localhost:3000
echo.
echo Both servers are now running in separate windows.
echo Close this window when you're done.
echo ================================================

pause
