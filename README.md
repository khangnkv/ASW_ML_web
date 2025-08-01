# ML Prediction System

## Setup (First Time)
1. Install Python 3.8+ and Node.js (https://nodejs.org/)
2. Run:

```bash
# Clone the repository
git clone <your-repo-url>
cd ASW_ML_web

# Setup Python environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install all dependencies and set up the system
python setup.py
```

## Features

- **File Upload**: CSV and Excel support (max 100K rows)
- **File Preview**: See first/last rows of uploaded data
- **Model Matching**: Auto-matches project_id to correct models
- **Predictions**: Appends predictions to original data
- **Export Options**: Download results as JSON, CSV, or Excel
- **Modern UI**: Responsive React frontend
- **Real-time Status**: Model and prediction stats

## System Architecture

```
ASW_ML_web/
├── backend/
│   ├── app_workflow.py         # Flask backend server
│   ├── model/
│   │   ├── predictor.py        # ML prediction logic
│   │   └── project_*_model.pkl # Pre-trained models
│   ├── uploads/                # Temporary file storage
│   └── preprocessed_unencoded/ # Preprocessed files
├── frontend/                   # React frontend
│   ├── package.json
│   ├── public/
│   └── src/
├── requirements.txt            # Python dependencies
├── setup.py                    # Python setup script
├── run_system.py               # Complete system runner
└── Dockerfile                  # Docker backend build
```

## Prerequisites

- Python 3.8+
- Node.js 16+
- npm

## Quick Start

### Option 1: Local (Recommended)

```bash
python setup.py
python run_system.py
```
- Backend: http://localhost:5000
- Frontend: http://localhost:3000

### Option 2: Docker

```bash
docker-compose up --build
```
- Backend: http://localhost:5000
- Frontend: http://localhost:3000

## Usage

1. **Upload File**: Must contain a `projectid` column.
2. **Preview Data**: See file info and preview.
3. **Generate Predictions**: Click to predict.
4. **Export Results**: Download as CSV, Excel, or JSON.

## API Endpoints

- `POST /api/upload` - Upload and preview file
- `POST /api/predict` - Generate predictions
- `GET /api/export/<format>/<filename>` - Export results
- `GET /api/models` - Get available models

## Data Requirements

- Input file must have a `projectid` column.
- Model files: `project_{project_id}_model.pkl` in `backend/model/`.

## Troubleshooting

- See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues.

## Debug Mode

- Toggle debug prints using `toggle_debug.py` or by editing `DEBUG_PRINTS` in relevant files.

## License

MIT License

## Support

- Open an issue on GitHub for help.