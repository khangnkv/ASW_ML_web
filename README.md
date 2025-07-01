# ML Prediction System

A Flask backend + React frontend system for uploading CSV/Excel files and generating predictions using pre-trained machine learning models.

## Features

- **File Upload**: Support for CSV and Excel files (max 100K rows)
- **File Preview**: Shows first and last 5-10 rows of uploaded data
- **Model Matching**: Automatically matches project_id to correct pre-trained models
- **Predictions**: Generates predictions and appends to original data
- **Export Options**: Export results as JSON, CSV, or Excel
- **Modern UI**: Beautiful, responsive React frontend with drag-and-drop upload
- **Real-time Status**: Shows model availability and prediction statistics

## System Architecture

```
web_dev/
├── app.py                 # Flask backend server
├── requirements.txt       # Python dependencies
├── setup.py              # Python setup script
├── run_system.py         # Complete system runner
├── start_backend.py      # Backend startup script
├── start_frontend.py     # Frontend startup script
├── model/
│   ├── predictor.py       # ML prediction logic
│   └── project_*_model.pkl # Pre-trained models
├── frontend/             # React frontend
│   ├── package.json
│   ├── public/
│   └── src/
└── uploads/              # Temporary file storage
```

## Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn

## Quick Start

### Option 1: One-Command Setup & Run (Recommended)

```bash
# Setup and run everything in one command
python run_system.py
```

This will:
- Check and install all dependencies
- Start both backend and frontend
- Open the application automatically

### Option 2: Manual Setup

1. **Setup the system:**
   ```bash
   python setup.py
   ```

2. **Run the complete system:**
   ```bash
   python run_system.py
   ```

### Option 3: Separate Backend & Frontend

1. **Start the backend:**
   ```bash
   python start_backend.py
   ```

2. **Start the frontend (in a new terminal):**
   ```bash
   python start_frontend.py
   ```

## Installation Details

### Backend Setup

The Python setup script (`setup.py`) will:
- Check Python version compatibility
- Install all required Python packages
- Create necessary directories
- Verify model files

### Frontend Setup

The setup script will also:
- Check Node.js installation
- Install npm dependencies
- Verify frontend structure

## Running the Application

### Development Mode

**Recommended: Use the complete system runner**
```bash
python run_system.py
```

**Alternative: Run components separately**
```bash
# Terminal 1: Backend
python start_backend.py

# Terminal 2: Frontend  
python start_frontend.py
```

### Production Mode

1. **Build the React frontend:**
   ```bash
   cd frontend
   npm run build
   ```

2. **Serve the built files with Flask:**
   ```bash
   python app.py
   ```

## 🚀 Running with Docker (Recommended for All Platforms)

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/Mac/Linux)
- [Docker Compose](https://docs.docker.com/compose/)

### Quick Start

1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd web_dev
   ```

2. **Build and start the system:**
   ```sh
   docker-compose up --build
   ```
   This will build both backend and frontend images and start them as containers.

3. **Access the system:**
   - Backend API: [http://localhost:5000](http://localhost:5000)
   - Frontend UI: [http://localhost:3000](http://localhost:3000)

4. **Stopping the system:**
   Press `Ctrl+C` in the terminal, then run:
   ```sh
   docker-compose down
   ```

### Notes
- Uploaded files and preprocessed data are persisted in the `backend/uploads` and `backend/preprocessed_unencoded` folders on your host machine.
- You can develop locally as usual; changes to code will require a rebuild (`docker-compose up --build`).
- For production, you may want to adjust environment variables and volumes as needed.

## Usage

1. **Upload File**: Drag and drop or click to upload a CSV/Excel file
   - File must contain a `projectid` column
   - Maximum 100,000 rows allowed
   - Supported formats: CSV, XLSX, XLS

2. **Preview Data**: The system shows:
   - File information (rows, columns, filename)
   - First and last 5 rows preview
   - Model availability for each project ID

3. **Generate Predictions**: Click "Generate Predictions" to:
   - Match each row's project_id to the correct model
   - Generate predictions for each row
   - Show prediction statistics

4. **Export Results**: Download results in:
   - CSV format
   - Excel format
   - JSON format

## API Endpoints

### Backend API

- `POST /api/upload` - Upload and preview file
- `POST /api/predict` - Generate predictions
- `GET /api/export/<format>/<filename>` - Export results
- `POST /api/cleanup` - Clean up uploaded files
- `GET /api/models` - Get available models

### Request/Response Examples

**Upload File:**
```bash
curl -X POST -F "file=@data.csv" http://localhost:5000/api/upload
```

**Generate Predictions:**
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"filename":"data.csv"}' \
  http://localhost:5000/api/predict
```

**Export Results:**
```bash
curl -X GET http://localhost:5000/api/export/csv/results_data.csv
```

## Data Requirements

### Input File Format

Your CSV/Excel file must contain:
- `projectid` column (required)
- Any additional feature columns used by your models

### Model File Format

Each model file should be saved as:
```
project_{project_id}_model.pkl
```

The pickle file should contain:
```python
{
    "model": trained_model_object,
    "features": ["feature1", "feature2", ...]  # optional
}
```

## Troubleshooting

### Common Issues

1. **"File must contain a projectid column"**
   - Ensure your CSV/Excel file has a column named exactly "projectid"

2. **"Model not found for project X"**
   - Check that `project_X_model.pkl` exists in the `model/` directory

3. **"File too large"**
   - Reduce file size to under 100,000 rows

4. **CORS errors**
   - Ensure the backend is running on the correct port
   - Check that CORS is properly configured

5. **Pandas installation issues**
   - The setup script uses flexible version requirements
   - If issues persist, try: `pip install pandas --upgrade`

### Debug Mode

Enable debug mode in Flask:
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

## Scripts Overview

| Script | Purpose | Usage |
|--------|---------|-------|
| `setup.py` | Install dependencies | `python setup.py` |
| `run_system.py` | Run complete system | `python run_system.py` |
| `start_backend.py` | Start backend only | `python start_backend.py` |
| `start_frontend.py` | Start frontend only | `python start_frontend.py` |
| `app.py` | Direct Flask app | `python app.py` |

## Security Considerations

- File size limits (100MB max)
- Row count limits (100K max)
- Temporary file cleanup
- Input validation
- CORS configuration

## Performance Optimization

- Efficient file reading with pandas
- Batch processing for large files
- Memory management for large datasets
- Asynchronous file operations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation
3. Open an issue on GitHub