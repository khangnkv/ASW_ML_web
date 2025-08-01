# Troubleshooting Guide

## Common Issues and Solutions

### 1. npm Not Found Error

**Problem:** `FileNotFoundError: [WinError 2] The system cannot find the file specified` when running `python run_system.py`

**Solutions:**

- Ensure Node.js is installed and in your PATH.
- Reinstall Node.js from https://nodejs.org/ (LTS version, "Add to PATH" checked).
- Verify with:
  ```
  node --version
  npm --version
  ```

### 2. Pandas Installation Issues

- Try: `pip install --only-binary=all pandas`
- Or: `pip install pandas --upgrade --force-reinstall`

### 3. Port Already in Use

- Find and kill process using port 5000:
  ```
  netstat -ano | findstr :5000
  taskkill /PID <PID> /F
  ```

### 4. Model Files Not Found

- Ensure `backend/model/` contains files like `project_6_model.pkl`.

### 5. CORS Errors in Browser

- Backend must be running on port 5000.
- Frontend must have `REACT_APP_API_URL` set to backend URL.

### 6. File Upload Issues

- File must be CSV/Excel with a `projectid` column.
- Max 100,000 rows, 100MB file size.

## Quick Test Commands

- Run full system: `python run_system.py`
- Test API endpoints with `curl` as shown in README.

## System Requirements

- Python 3.8+
- Node.js 16+
- At least 4GB RAM, 1GB disk

## Getting Help

- Check this guide and README.
- Run with verbose output: `python -v run_system.py`
- Check logs in terminal and browser console.