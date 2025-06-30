@echo off
title ML Prediction System - Backend Server

echo ================================================
echo   ML Prediction System - Backend Server
echo ================================================
echo Starting backend server...
echo.

REM Change to the project root directory
cd /d "%~dp0"

REM Start the backend from the correct directory structure
python backend/app_workflow.py

echo.
echo Backend server stopped.
pause
