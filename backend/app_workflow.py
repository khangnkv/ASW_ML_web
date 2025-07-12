import uuid
import json
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import sys
import os
import io
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.model.predictor import MLPredictor
from preprocessing import preprocess_data, get_raw_preview
from backend.data_retention import retention_manager

# Debug configuration - set to False to disable debug prints
DEBUG_PRINTS = True

app = Flask(__name__)
# Enhanced CORS configuration
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:3000", 
            "http://127.0.0.1:3000",
            "http://frontend:3000",
            "http://ml-frontend:3000"
        ],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Always resolve uploads and preprocessed dirs relative to this file's directory
BACKEND_DIR = Path(__file__).parent
UPLOADS_DIR = BACKEND_DIR / 'uploads'
PREPROCESSED_DIR = BACKEND_DIR / 'preprocessed_unencoded'
PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

predictor = MLPredictor()

def sanitize_preview(preview):
    def safe(val):
        if pd.isna(val):
            return ""
        if isinstance(val, (pd.Timestamp, datetime)):
            return str(val)
        return val
    return [{k: safe(v) for k, v in row.items()} for row in preview]

def sanitize_records(records):
    def safe(val):
        if pd.isna(val):
            return ""
        if isinstance(val, (pd.Timestamp, datetime)):
            return str(val)
        return val
    return [{k: safe(v) for k, v in row.items()} for row in records]

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
        model_dir = Path('../model')
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
    """Export results in specified format with enhanced functionality"""
    try:
        if DEBUG_PRINTS:
            print(f"[EXPORT] Exporting {filename} as {format}")
        
        # Look for the file in uploads directory
        file_path = UPLOADS_DIR / filename
        if not file_path.exists():
            if DEBUG_PRINTS:
                print(f"[EXPORT] File not found at {file_path}")
            return jsonify({'error': 'File not found'}), 404
        
        # Read the data
        try:
            df = pd.read_csv(file_path)
            if DEBUG_PRINTS:
                print(f"[EXPORT] Loaded data with {len(df)} rows and {len(df.columns)} columns")
        except Exception as e:
            if DEBUG_PRINTS:
                print(f"[EXPORT] Error reading file: {e}")
            return jsonify({'error': f'Error reading file: {str(e)}'}), 500
        
        # Get original filename without UUID for cleaner export names
        original_name = filename.replace('.csv', '')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'json':
            # Convert DataFrame to JSON with better formatting
            try:
                # Convert to records format and handle NaN values
                result = df.fillna('').to_dict(orient='records')
                
                # Create response with proper filename
                response_data = {
                    'data': result,
                    'metadata': {
                        'total_records': len(result),
                        'columns': list(df.columns),
                        'export_timestamp': datetime.now().isoformat(),
                        'format': 'json'
                    }
                }
                
                response = jsonify(response_data)
                response.headers['Content-Disposition'] = f'attachment; filename=predictions_{original_name}_{timestamp}.json'
                return response
                
            except Exception as e:
                if DEBUG_PRINTS:
                    print(f"[EXPORT] Error creating JSON: {e}")
                return jsonify({'error': f'Error creating JSON: {str(e)}'}), 500
                
        elif format == 'csv':
            # Return CSV file with proper encoding
            try:
                output = io.StringIO()
                df.to_csv(output, index=False, encoding='utf-8')
                output.seek(0)
                
                return send_file(
                    io.BytesIO(output.getvalue().encode('utf-8-sig')),  # UTF-8 with BOM for Excel compatibility
                    mimetype='text/csv',
                    as_attachment=True,
                    download_name=f'predictions_{original_name}_{timestamp}.csv'
                )
            except Exception as e:
                if DEBUG_PRINTS:
                    print(f"[EXPORT] Error creating CSV: {e}")
                return jsonify({'error': f'Error creating CSV: {str(e)}'}), 500
                
        elif format == 'xlsx':
            # Return Excel file with enhanced formatting
            try:
                output = io.BytesIO()
                
                with pd.ExcelWriter(output, engine='openpyxl', mode='w') as writer:
                    # Write main data
                    df.to_excel(writer, index=False, sheet_name='Predictions')
                    
                    # Add a summary sheet if we have prediction data
                    if 'has_booked_prediction' in df.columns:
                        summary_data = {
                            'Metric': [
                                'Total Records',
                                'Records with Predictions',
                                'Predicted Positive (Likely to Book)',
                                'Predicted Negative (Unlikely to Book)',
                                'Prediction Rate (%)',
                                'Positive Prediction Rate (%)',
                                'Export Date'
                            ],
                            'Value': [
                                len(df),
                                len(df[df['has_booked_prediction'].notna()]),
                                len(df[df['has_booked_prediction'] >= 0.5]),
                                len(df[df['has_booked_prediction'] < 0.5]),
                                f"{(len(df[df['has_booked_prediction'].notna()]) / len(df) * 100):.1f}%",
                                f"{(len(df[df['has_booked_prediction'] >= 0.5]) / len(df[df['has_booked_prediction'].notna()]) * 100):.1f}%" if len(df[df['has_booked_prediction'].notna()]) > 0 else "N/A",
                                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            ]
                        }
                        summary_df = pd.DataFrame(summary_data)
                        summary_df.to_excel(writer, index=False, sheet_name='Summary')
                    
                    # Auto-adjust column widths
                    for sheet_name in writer.sheets:
                        worksheet = writer.sheets[sheet_name]
                        for column in worksheet.columns:
                            max_length = 0
                            column_letter = column[0].column_letter
                            for cell in column:
                                try:
                                    if len(str(cell.value)) > max_length:
                                        max_length = len(str(cell.value))
                                except:
                                    pass
                            adjusted_width = min(max_length + 2, 50)  # Cap at 50 characters
                            worksheet.column_dimensions[column_letter].width = adjusted_width
                
                output.seek(0)
                
                return send_file(
                    output,
                    mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    as_attachment=True,
                    download_name=f'predictions_{original_name}_{timestamp}.xlsx'
                )
            except Exception as e:
                if DEBUG_PRINTS:
                    print(f"[EXPORT] Error creating Excel: {e}")
                return jsonify({'error': f'Error creating Excel: {str(e)}'}), 500
        else:
            return jsonify({'error': f'Unsupported export format: {format}'}), 400
            
    except Exception as e:
        if DEBUG_PRINTS:
            print(f"[EXPORT] General error: {e}")
            import traceback
            traceback.print_exc()
        return jsonify({'error': f'Error exporting results: {str(e)}'}), 500

# Helper: Save uploaded file with UUID
@app.route('/api/upload_uuid', methods=['POST'])
def upload_uuid():
    print("[UPLOAD] /api/upload_uuid endpoint called")
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        file = request.files['file']
        print(f"[UPLOAD] Received file: {getattr(file, 'filename', None)}")
        if not file or not file.filename:
            return jsonify({'error': 'No selected file'}), 400

        ext = file.filename.rsplit('.', 1)[-1].lower()
        if ext not in ['csv', 'xlsx', 'xls']:
            return jsonify({'error': 'Unsupported file type'}), 400

        # Read file into DataFrame
        if ext == 'csv':
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        # Ensure 'projectid' column exists
        if 'projectid' not in df.columns:
            # Try to auto-rename common alternatives
            if 'Project ID' in df.columns:
                df.rename(columns={'Project ID': 'projectid'}, inplace=True)
            else:
                return jsonify({'error': "File must contain a 'projectid' column"}), 400

        # Save file with UUID
        uuid_name = f"{uuid.uuid4()}.csv"
        save_path = UPLOADS_DIR / uuid_name
        df.to_csv(save_path, index=False)
        print(f"[UPLOAD] File saved to: {save_path.resolve()}")

        # Sanitize for JSON serialization
        def safe(val):
            if pd.isna(val):
                return ""
            if isinstance(val, (pd.Timestamp, datetime)):
                return str(val)
            return val

        preview = [{k: safe(v) for k, v in row.items()} for row in df.head(5).to_dict(orient='records')]
        full_dataset = [{k: safe(v) for k, v in row.items()} for row in df.to_dict(orient='records')]

        # File info for retention
        now = datetime.utcnow()
        retention_days = 90
        file_info = {
            "filename": uuid_name,
            "originalName": file.filename,
            "upload_timestamp": now.isoformat(),
            "deletion_date": (now + timedelta(days=retention_days)).isoformat(),
            "status": "active"
        }

        response_data = {
            "filename": uuid_name,
            "preview": preview,
            "full_dataset": full_dataset,
            "file_info": file_info
        }

        return jsonify(response_data)
        
    except Exception as e:
        print(f"[UPLOAD][ERROR] {e}")
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

# Main workflow endpoint
@app.route('/api/predict_workflow', methods=['POST'])
def predict_workflow():
    if DEBUG_PRINTS:
        print("[PREDICT] /api/predict_workflow endpoint called")
    try:
        data = request.get_json()
        if not data or 'filename' not in data:
            return jsonify({'error': 'No filename provided'}), 400
        
        filename = data['filename']
        upload_path = UPLOADS_DIR / filename
        
        if not upload_path.exists():
            return jsonify({'error': f'File {filename} not found'}), 404
        
        if DEBUG_PRINTS:
            print(f"[PREDICT] Processing file: {upload_path}")
        
        # 1. Read raw data
        if filename.endswith('.csv'):
            raw_df = pd.read_csv(upload_path)
        else:
            raw_df = pd.read_excel(upload_path)
        
        if DEBUG_PRINTS:
            print(f"[PREDICT] Raw data shape: {raw_df.shape}")
        
        # 2. Preprocessing
        try:
            processed_df = preprocess_data(str(upload_path))
            if not isinstance(processed_df, pd.DataFrame):
                processed_df = pd.DataFrame(processed_df)
            if DEBUG_PRINTS:
                print(f"[PREDICT] Processed data shape: {processed_df.shape}")
        except Exception as e:
            if DEBUG_PRINTS:
                print(f"[PREDICT] Preprocessing error: {e}")
            return jsonify({'error': f'Preprocessing failed: {str(e)}'}), 500
        
        # 3. Generate predictions using the main predict method
        try:
            # Use the main predict method which handles all projects at once
            result_df = predictor.predict(processed_df, log_progress=True)
            
            # Add confidence scores
            result_df = predictor.add_prediction_confidence(result_df)
            
            # Extract predictions from the result
            predictions_list = []
            prediction_counts = {'likely': 0, 'unlikely': 0}
            
            if 'has_booked_prediction' in result_df.columns:
                for idx, row in result_df.iterrows():
                    pred_value = row['has_booked_prediction']
                    confidence = row.get('prediction_confidence', 0.95)
                    if not pd.isna(pred_value):
                        predictions_list.append({
                            'projectid': int(row['projectid']),
                            'row_index': idx,
                            'prediction': float(pred_value),
                            'confidence': float(confidence)
                        })
                        # Count predictions
                        if pred_value >= 0.5:
                            prediction_counts['likely'] += 1
                        else:
                            prediction_counts['unlikely'] += 1
            
            if DEBUG_PRINTS:
                print(f"[PREDICT] Generated {len(predictions_list)} predictions")
            
            # Return the complete dataset with predictions for the frontend table
            complete_dataset = sanitize_records(result_df.to_dict(orient='records'))
            
            response_data = {
                'predictions': predictions_list,
                'complete_dataset': complete_dataset,
                'raw_preview': sanitize_records(raw_df.head(5).to_dict(orient='records')),
                'processed_preview': sanitize_records(processed_df.head(5).to_dict(orient='records')),
                'prediction_counts': prediction_counts,
                'summary': {
                    'total_rows': len(processed_df),
                    'projects_processed': len(processed_df['projectid'].unique()) if 'projectid' in processed_df.columns else 0,
                    'predictions_generated': len(predictions_list)
                }
            }
            
            return jsonify(response_data)
            
        except Exception as e:
            if DEBUG_PRINTS:
                print(f"[PREDICT] Prediction error: {e}")
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
            
    except Exception as e:
        if DEBUG_PRINTS:
            print(f"[PREDICT][ERROR] {e}")
            import traceback
            traceback.print_exc()
        return jsonify({'error': f'Error processing request: {str(e)}'}), 500

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
    print("=" * 50)
    print("  ML Prediction System Backend")
    print("=" * 50)
    print("Backend server starting...")
    print("Health check: http://localhost:5000/api/health")
    print("Models endpoint: http://localhost:5000/api/models")
    print("=" * 50)
    
    # Run in production mode to prevent auto-reload issues
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)