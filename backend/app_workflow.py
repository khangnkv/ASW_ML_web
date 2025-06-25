import uuid
import json
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from model.predictor import MLPredictor
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import preprocess_data, get_raw_preview

app = Flask(__name__)
CORS(app)

UPLOADS_DIR = Path('uploads')
PREPROCESSED_DIR = Path('backend/preprocessed_unencoded')
PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

predictor = MLPredictor()

def sanitize_preview(preview):
    def safe(val):
        if pd.isna(val):
            return ""
        if isinstance(val, (pd.Timestamp, datetime)):
            return str(val)
        return val
    return [{k: safe(v) for k, v in row.items()} for row in preview]

# Helper: Save uploaded file with UUID
@app.route('/api/upload_uuid', methods=['POST'])
def upload_uuid():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if not file or not file.filename:
        return jsonify({'error': 'No file selected'}), 400
    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in {'csv', 'xlsx', 'xls'}:
        return jsonify({'error': 'Invalid file type'}), 400
    uuid_name = f"{uuid.uuid4()}.{ext}"
    save_path = UPLOADS_DIR / uuid_name
    file.save(save_path)
    # Read for preview
    try:
        if ext == 'csv':
            df = pd.read_csv(save_path)
        else:
            df = pd.read_excel(save_path)
        preview = get_raw_preview(df, n=5).to_dict(orient='records')
        preview = sanitize_preview(preview)
        print("File saved to:", save_path)
        print("DataFrame shape:", df.shape)
        print("Preview data:", preview)
    except Exception as e:
        return jsonify({'error': f'Error reading file: {e}'}), 400
    return jsonify({'filename': uuid_name, 'preview': preview})

# Main workflow endpoint
@app.route('/api/predict_workflow', methods=['POST'])
def predict_workflow():
    data = request.json
    filename = data.get('filename')
    if not filename:
        return jsonify({'error': 'Filename not provided'}), 400
    upload_path = UPLOADS_DIR / filename
    if not upload_path.exists():
        return jsonify({'error': 'File not found'}), 404
    try:
        # 1. Raw preview
        raw_df = pd.read_csv(upload_path) if filename.endswith('.csv') else pd.read_excel(upload_path)
        raw_preview = get_raw_preview(raw_df, n=5).to_dict(orient='records')
        raw_preview = sanitize_preview(raw_preview)
        # 2. Preprocessing (25%)
        processed_df = preprocess_data(str(upload_path))
        processed_preview = get_raw_preview(processed_df, n=5).to_dict(orient='records')
        processed_preview = sanitize_preview(processed_preview)
        # 3. Prediction (50%)
        pred_df = predictor.predict(processed_df, log_progress=False)
        final_preview = get_raw_preview(pred_df, n=5).to_dict(orient='records')
        final_preview = sanitize_preview(final_preview)
        # 4. Stats
        pred_col = pred_df['has_booked_prediction']
        pred_dist = pred_col.value_counts(dropna=False).to_dict()
        stats = {
            'rows_processed': int(len(pred_df)),
            'prediction_distribution': pred_dist
        }
        # 5. Export formats
        export_formats = ['csv', 'xlsx', 'json']
        # 6. Save final results for export
        results_filename = f"results_{filename.rsplit('.', 1)[0]}.csv"
        results_path = UPLOADS_DIR / results_filename
        pred_df.to_csv(results_path, index=False)
        # 7. Response
        return jsonify({
            'preview': {
                'raw': raw_preview,
                'processed': processed_preview,
                'final': final_preview
            },
            'export_formats': export_formats,
            'stats': stats,
            'results_filename': results_filename
        })
    except Exception as e:
        # Clean up temp files on error
        return jsonify({'error': f'Workflow error: {e}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)