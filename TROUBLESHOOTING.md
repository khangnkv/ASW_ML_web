# Troubleshooting Guide

## Common Issues and Solutions

### 1. npm Not Found Error

**Problem:** `FileNotFoundError: [WinError 2] The system cannot find the file specified` when running `python run_system.py`

**Solutions:**

#### Option A: Fix npm PATH (Recommended)
1. **Reinstall Node.js:**
   - Download from https://nodejs.org/
   - Choose the LTS version
   - During installation, make sure "Add to PATH" is checked

2. **Verify installation:**
   ```bash
   node --version
   npm --version
   ```

3. **If still not working, manually add to PATH:**
   - Open System Properties → Environment Variables
   - Add these to PATH:
     ```
     C:\Program Files\nodejs\
     %APPDATA%\npm
     ```

#### Option B: Use Backend Only (Quick Fix)
If you just want to test the backend functionality:

```bash
python run_backend_only.py
```

This will start just the Flask backend at http://localhost:5000

#### Option C: Manual Frontend Setup
1. **Open Command Prompt as Administrator**
2. **Navigate to frontend directory:**
   ```bash
   cd frontend
   ```
3. **Install dependencies:**
   ```bash
   npm install
   ```
4. **Start frontend:**
   ```bash
   npm start
   ```

### 2. Pandas Installation Issues

**Problem:** Build errors when installing pandas

**Solutions:**

#### Option A: Use Pre-built Wheels
```bash
pip install --only-binary=all pandas
```

#### Option B: Install from Conda (if available)
```bash
conda install pandas
```

#### Option C: Use Alternative Installation
```bash
pip install pandas --upgrade --force-reinstall
```

### 3. Port Already in Use

**Problem:** `Address already in use` error

**Solutions:**

#### Option A: Kill Process on Port
```bash
# Find process using port 5000
netstat -ano | findstr :5000

# Kill the process (replace PID with actual process ID)
taskkill /PID <PID> /F
```

#### Option B: Use Different Port
Edit `app.py` and change the port:
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

### 4. Model Files Not Found

**Problem:** "No model files found" warning

**Solutions:**

1. **Check model directory structure:**
   ```
   model/
   ├── project_6_model.pkl
   ├── project_8_model.pkl
   └── project_10_model.pkl
   ```

2. **Verify file naming convention:**
   - Must be: `project_{id}_model.pkl`
   - Example: `project_6_model.pkl`

3. **Test with sample data:**
   - Use the provided `sample_data.csv`
   - Contains project IDs: 6, 8, 10, 12, 13

### 5. CORS Errors in Browser

**Problem:** Frontend can't connect to backend

**Solutions:**

1. **Check backend is running:**
   ```bash
   curl http://localhost:5000/api/models
   ```

2. **Verify frontend proxy setting:**
   In `frontend/package.json`, ensure:
   ```json
   "proxy": "http://localhost:5000"
   ```

3. **Check browser console for errors**

### 6. File Upload Issues

**Problem:** File upload fails

**Solutions:**

1. **Check file format:**
   - Must be CSV or Excel
   - Must contain `projectid` column

2. **Check file size:**
   - Maximum 100,000 rows
   - Maximum 100MB file size

3. **Test with sample file:**
   ```bash
   curl -X POST -F "file=@sample_data.csv" http://localhost:5000/api/upload
   ```

## Quick Test Commands

### Test Backend Only
```bash
python run_backend_only.py
```

### Test API Endpoints
```bash
# Get available models
curl http://localhost:5000/api/models

# Upload file
curl -X POST -F "file=@sample_data.csv" http://localhost:5000/api/upload

# Generate predictions
curl -X POST -H "Content-Type: application/json" -d '{"filename":"sample_data.csv"}' http://localhost:5000/api/predict
```

### Test Frontend Only
```bash
cd frontend
npm install
npm start
```

## System Requirements

- **Python:** 3.8 or higher
- **Node.js:** 16 or higher
- **npm:** Comes with Node.js
- **Memory:** At least 4GB RAM
- **Disk:** At least 1GB free space

## Getting Help

1. **Check this troubleshooting guide**
2. **Run with verbose output:**
   ```bash
   python -v run_system.py
   ```
3. **Check logs:**
   - Backend logs appear in terminal
   - Frontend logs appear in browser console
4. **Test components separately** using the individual scripts

## Alternative Setup Methods

### Method 1: Backend Only (No Frontend)
```bash
python run_backend_only.py
```
Then use curl, Postman, or any HTTP client to test the API.

### Method 2: Manual Component Setup
```bash
# Terminal 1: Backend
python app.py

# Terminal 2: Frontend
cd frontend
npm install
npm start
```

### Method 3: Docker (Advanced)
If you have Docker installed, you can containerize the application (requires additional setup). 