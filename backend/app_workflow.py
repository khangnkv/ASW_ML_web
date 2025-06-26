import uuid
import json
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
from model.predictor import MLPredictor
import sys
import os
import io
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import preprocess_data, get_raw_preview
from data_retention import retention_manager

app = Flask(__name__)
# Enhanced CORS configuration
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

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

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Backend is running'})

# Models endpoint
@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models"""
    try:
        model_dir = Path('model')
        model_files = list(model_dir.glob('*.pkl'))
        available_models = [f.stem for f in model_files]
        return jsonify({
            'available_models': available_models,
            'total_models': len(available_models)
        })
    except Exception as e:
        return jsonify({'error': f'Error getting models: {e}'}), 500

# Export endpoint
@app.route('/api/export/<format>/<filename>', methods=['GET'])
def export_results(format, filename):
    """Export results in specified format"""
    try:
        file_path = UPLOADS_DIR / filename
        if not file_path.exists():
            return jsonify({'error': 'File not found'}), 404
        
        df = pd.read_csv(file_path)
        
        if format == 'json':
            # Convert DataFrame to JSON
            result = df.to_dict(orient='records')
            return jsonify(result)
        elif format == 'csv':
            # Return CSV file
            output = io.StringIO()
            df.to_csv(output, index=False)
            output.seek(0)
            return send_file(
                io.BytesIO(output.getvalue().encode('utf-8')),
                mimetype='text/csv',
                as_attachment=True,
                download_name=f'predictions_{filename}'
            )
        elif format == 'xlsx':
            # Return Excel file
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Predictions')
            output.seek(0)
            return send_file(
                output,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                as_attachment=True,
                download_name=f'predictions_{filename.replace(".csv", ".xlsx")}'
            )
        else:
            return jsonify({'error': 'Unsupported export format'}), 400
    except Exception as e:
        return jsonify({'error': f'Error exporting results: {e}'}), 500

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
    
    # Always save as CSV for faster preprocessing
    uuid_name = f"{uuid.uuid4()}.csv"
    save_path = UPLOADS_DIR / uuid_name
    
    # Read the uploaded file and convert to CSV
    try:
        if ext == 'csv':
            df = pd.read_csv(file.stream)
        else:
            df = pd.read_excel(file.stream, engine='openpyxl' if ext == 'xlsx' else 'xlrd')
        
        # Save as CSV for faster preprocessing
        df.to_csv(save_path, index=False)
        
        # Store file metadata with retention manager
        file_info = retention_manager.store_file(uuid_name, file.filename)
        
        preview = get_raw_preview(df, n=5).to_dict(orient='records')
        preview = sanitize_preview(preview)
        print("File saved to:", save_path)
        print("DataFrame shape:", df.shape)
        print("Preview data:", preview)
        
        return jsonify({
            'filename': uuid_name, 
            'preview': preview,
            'file_info': file_info
        })
    except Exception as e:
        return jsonify({'error': f'Error reading file: {e}'}), 400

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
        # --- Feature validation and logging ---
        projectids = processed_df['projectid'].unique()
        missing_features_report = {}
        for pid in projectids:
            try:
                pid_int = int(pid)
            except Exception:
                continue
            if pid_int in predictor.models:
                _, features = predictor.models[pid_int]
                missing = [f for f in features if f not in processed_df.columns]
                if missing:
                    missing_features_report[pid_int] = missing
        if missing_features_report:
            return jsonify({'error': f'Missing required features for prediction', 'details': missing_features_report, 'columns': list(processed_df.columns)}), 400
        print('Columns before prediction:', list(processed_df.columns))
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
        import traceback
        tb = traceback.format_exc()
        return jsonify({'error': f'Workflow error: {e}', 'traceback': tb}), 500

# Data retention management endpoints
@app.route('/api/storage/stats', methods=['GET'])
def get_storage_stats():
    """Get storage statistics"""
    try:
        stats = retention_manager.get_storage_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': f'Error getting storage stats: {e}'}), 500

@app.route('/api/storage/files', methods=['GET'])
def list_stored_files():
    """List all stored files with metadata"""
    try:
        files = retention_manager.list_files()
        return jsonify({'files': files})
    except Exception as e:
        return jsonify({'error': f'Error listing files: {e}'}), 500

@app.route('/api/storage/files/<filename>', methods=['DELETE'])
def delete_stored_file(filename):
    """Manually delete a stored file"""
    try:
        success = retention_manager.delete_file(filename)
        if success:
            return jsonify({'message': f'File {filename} deleted successfully'})
        else:
            return jsonify({'error': f'File {filename} not found'}), 404
    except Exception as e:
        return jsonify({'error': f'Error deleting file: {e}'}), 500

@app.route('/api/storage/cleanup', methods=['POST'])
def manual_cleanup():
    """Manually trigger cleanup of expired files"""
    try:
        deleted_count = retention_manager.cleanup_expired_files()
        return jsonify({
            'message': f'Cleanup completed',
            'deleted_count': deleted_count
        })
    except Exception as e:
        return jsonify({'error': f'Error during cleanup: {e}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)