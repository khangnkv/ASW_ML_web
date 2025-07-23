import uuid
import json
from datetime import datetime, timedelta
from pathlib import Path
import warnings
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import pandas as pd
import sys
import os
import io
import logging

# Suppress warnings globally
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', message='.*version.*when using version.*')
warnings.filterwarnings('ignore', message='.*InconsistentVersionWarning.*')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['PYTHONWARNINGS'] = 'ignore'

# Configure logging
logging.basicConfig(level=logging.DEBUG)

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

# Health check endpoint with more detailed diagnostics
@app.route('/api/health', methods=['GET'])
def health_check():
    """Enhanced health check endpoint"""
    try:
        health_status = {
            'status': 'healthy',
            'message': 'Backend is running',
            'timestamp': datetime.now().isoformat(),
            'models_loaded': len(predictor.models),
            'available_models': list(predictor.models.keys()) if hasattr(predictor, 'models') else [],
            'uploads_dir_exists': UPLOADS_DIR.exists(),
            'predictor_initialized': predictor is not None
        }
        
        # Test basic predictor functionality
        if hasattr(predictor, 'models') and len(predictor.models) > 0:
            health_status['model_status'] = 'ready'
        else:
            health_status['model_status'] = 'no_models_loaded'
            health_status['status'] = 'degraded'
        
        return jsonify(health_status)
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'message': f'Health check failed: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

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
def upload_file_uuid():
    print(f"[UPLOAD] Request method: {request.method}")
    print(f"[UPLOAD] Request files: {list(request.files.keys())}")
    print(f"[UPLOAD] Request headers: {dict(request.headers)}")
    
    if DEBUG_PRINTS:
        print("[UPLOAD] /api/upload_uuid endpoint called")
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if DEBUG_PRINTS:
            print(f"[UPLOAD] Received file: {file.filename}")
        
        # Generate UUID for the file
        file_uuid = str(uuid.uuid4())
        
        # Save file with UUID name but preserve original extension for processing
        original_filename = file.filename
        file_extension = Path(original_filename).suffix.lower()
        
        # Always save as CSV for consistency, but read original format first
        temp_filename = f"{file_uuid}{file_extension}"
        temp_path = UPLOADS_DIR / temp_filename
        file.save(temp_path)
        
        if DEBUG_PRINTS:
            print(f"[UPLOAD] Temporary file saved to: {temp_path}")
        
        # Read the file based on its extension
        try:
            if file_extension == '.csv':
                df = pd.read_csv(temp_path, low_memory=False)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(temp_path)
            else:
                return jsonify({'error': f'Unsupported file format: {file_extension}'}), 400
            
            # Convert to CSV for consistent processing
            final_filename = f"{file_uuid}.csv"
            final_path = UPLOADS_DIR / final_filename
            df.to_csv(final_path, index=False)
            
            # Remove temporary file if it's different from final
            if temp_path != final_path and temp_path.exists():
                temp_path.unlink()
            
            if DEBUG_PRINTS:
                print(f"[UPLOAD] File saved to: {final_path}")
                print(f"[UPLOAD] Data shape: {df.shape}")
                print(f"[UPLOAD] Columns: {list(df.columns)}")
            
        except Exception as e:
            if DEBUG_PRINTS:
                print(f"[UPLOAD] Error reading file: {e}")
            # Clean up temporary file
            if temp_path.exists():
                temp_path.unlink()
            return jsonify({'error': f'Error reading file: {str(e)}'}), 400
        
        # Validate required columns
        if 'projectid' not in df.columns:
            return jsonify({'error': 'File must contain a "projectid" column'}), 400
        
        # Create file info for tracking
        file_info = {
            'filename': final_filename,
            'original_name': original_filename,
            'uuid': file_uuid,
            'upload_timestamp': datetime.now().isoformat(),
            'deletion_date': (datetime.now() + timedelta(days=90)).isoformat(),
            'status': 'uploaded',
            'size_bytes': final_path.stat().st_size,
            'rows': len(df),
            'columns': len(df.columns)
        }
        
        # Generate preview data (first and last few rows)
        preview_df = get_raw_preview(df, n=5)
        preview_records = sanitize_preview(preview_df.to_dict(orient='records'))
        
        # Convert full dataset to records
        full_dataset = sanitize_records(df.to_dict(orient='records'))
        
        # IMPORTANT: Return the exact structure the frontend expects
        response_data = {
            'message': 'File uploaded successfully',
            'filename': final_filename,
            'original_name': original_filename,
            'uuid': file_uuid,
            'preview': preview_records,  # Frontend expects this
            'full_dataset': full_dataset,  # Frontend expects this
            'file_info': file_info,  # Frontend expects this
            'success': True
        }
        
        if DEBUG_PRINTS:
            print(f"[UPLOAD] Response structure: {list(response_data.keys())}")
            print(f"[UPLOAD] Preview rows: {len(preview_records)}")
            print(f"[UPLOAD] Full dataset rows: {len(full_dataset)}")
        
        return jsonify(response_data)
        
    except Exception as e:
        if DEBUG_PRINTS:
            print(f"[UPLOAD] Upload error: {e}")
            import traceback
            traceback.print_exc()
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

# Main workflow endpoint
@app.route('/api/predict_workflow', methods=['POST'])
def predict_workflow():
    """Complete workflow: Upload -> Preprocess -> Predict -> Return results"""
    if DEBUG_PRINTS:
        print(f"[PREDICT] /api/predict_workflow endpoint called")
        print(f"[PREDICT] Current working directory: {os.getcwd()}")
        print(f"[PREDICT] Script location: {Path(__file__).parent}")
        
        # Check if ProjectID_Detail.xlsx exists in expected locations
        test_paths = [
            '/app/backend/notebooks/project_info/ProjectID_Detail.xlsx',
            '/app/notebooks/project_info/ProjectID_Detail.xlsx',
            '/app/project_info/ProjectID_Detail.xlsx'
        ]
        for test_path in test_paths:
            exists = Path(test_path).exists()
            print(f"[PREDICT] Path check: {test_path} - Exists: {exists}")
    
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
        
        # 1. Read raw data with error handling
        try:
            if filename.endswith('.csv'):
                raw_df = pd.read_csv(upload_path, low_memory=False)
            else:
                raw_df = pd.read_excel(upload_path)
            
            if DEBUG_PRINTS:
                print(f"[PREDICT] Raw data shape: {raw_df.shape}")
        except Exception as e:
            if DEBUG_PRINTS:
                print(f"[PREDICT] Error reading file: {e}")
            return jsonify({'error': f'Error reading file: {str(e)}'}), 500
        
        # 2. Preprocessing with enhanced error handling
        try:
            processed_df = preprocess_data(str(upload_path))
            if not isinstance(processed_df, pd.DataFrame):
                processed_df = pd.DataFrame(processed_df)
            if DEBUG_PRINTS:
                print(f"[PREDICT] Processed data shape: {processed_df.shape}")
        except Exception as e:
            if DEBUG_PRINTS:
                print(f"[PREDICT] Preprocessing error: {e}")
                import traceback
                traceback.print_exc()
            return jsonify({'error': f'Preprocessing failed: {str(e)}'}), 500
        
        # 3. Check if predictor has models
        if not hasattr(predictor, 'models') or len(predictor.models) == 0:
            return jsonify({'error': 'No ML models available for prediction'}), 500
        
        # 4. Generate predictions with enhanced error handling
        try:
            # Suppress warnings during prediction
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result_df = predictor.predict(processed_df, log_progress=True)
                result_df = predictor.add_prediction_confidence(result_df)
            
            if DEBUG_PRINTS:
                print(f"[PREDICT] Result data shape: {result_df.shape}")
                print(f"[PREDICT] Result columns: {list(result_df.columns)}")
            
            # 5. Save the PROCESSED data with predictions (for analysis)
            processed_filename = filename.replace('.csv', '_processed_with_predictions.csv')
            processed_path = UPLOADS_DIR / processed_filename
            result_df.to_csv(processed_path, index=False)
            
            if DEBUG_PRINTS:
                print(f"[PREDICT] Saved processed data with predictions to: {processed_path}")
            
            # 6. Create prediction_results directory
            prediction_results_dir = UPLOADS_DIR / "prediction_results"
            prediction_results_dir.mkdir(exist_ok=True)
            
            # 7. IMPROVED: Map predictions back to raw data for frontend display
            raw_with_predictions = raw_df.copy()
            raw_with_predictions['has_booked_prediction'] = np.nan
            raw_with_predictions['prediction_confidence'] = 0.5
            
            # Create prediction mapping by projectid - IMPROVED LOGIC
            if 'projectid' in raw_df.columns and 'projectid' in result_df.columns:
                if DEBUG_PRINTS:
                    print(f"[PREDICT] Mapping predictions from processed data to raw data")
                    print(f"[PREDICT] Raw data projects: {sorted(raw_df['projectid'].unique())}")
                    print(f"[PREDICT] Processed data projects: {sorted(result_df['projectid'].unique())}")
                
                # Create a more robust mapping using both index and projectid
                prediction_map = {}
                
                # Build prediction dictionary from result_df
                for idx, row in result_df.iterrows():
                    project_id = row['projectid']
                    prediction = row.get('has_booked_prediction', np.nan)
                    confidence = row.get('prediction_confidence', 0.5)
                    
                    # Skip rows without valid predictions
                    if pd.isna(prediction):
                        continue
                        
                    # Create a key that can match back to raw data
                    key = f"{project_id}"
                    
                    if key not in prediction_map:
                        prediction_map[key] = []
                        
                    prediction_map[key].append({
                        'prediction': prediction,
                        'confidence': confidence,
                        'original_index': idx
                    })
                
                if DEBUG_PRINTS:
                    print(f"[PREDICT] Created prediction map with {len(prediction_map)} project groups")
                    print(f"[PREDICT] Total predictions available: {sum(len(v) for v in prediction_map.values())}")
                
                # Map predictions back to raw data
                mapped_count = 0
                for idx, row in raw_with_predictions.iterrows():
                    project_id = row['projectid']
                    key = f"{project_id}"
                    
                    if key in prediction_map and prediction_map[key]:
                        # Take the first available prediction for this project
                        pred_data = prediction_map[key].pop(0)
                        raw_with_predictions.at[idx, 'has_booked_prediction'] = pred_data['prediction']
                        raw_with_predictions.at[idx, 'prediction_confidence'] = pred_data['confidence']
                        mapped_count += 1
                
                if DEBUG_PRINTS:
                    print(f"[PREDICT] Successfully mapped {mapped_count} predictions to raw data")
                    print(f"[PREDICT] Predictions with valid values: {raw_with_predictions['has_booked_prediction'].notna().sum()}")

            else:
                if DEBUG_PRINTS:
                    print(f"[PREDICT] Warning: projectid column missing in raw or processed data")
                    print(f"[PREDICT] Raw columns: {list(raw_df.columns)}")
                    print(f"[PREDICT] Processed columns: {list(result_df.columns)}")
        
            # 8. Save prediction results (raw + predictions) to separate folder
            prediction_results_filename = filename.replace('.csv', '_prediction_results.csv')
            prediction_results_path = prediction_results_dir / prediction_results_filename
            raw_with_predictions.to_csv(prediction_results_path, index=False)
            
            if DEBUG_PRINTS:
                print(f"[PREDICT] Saved prediction results to: {prediction_results_path}")
            
            # 9. Extract predictions summary for response
            predictions_list = []
            prediction_counts = {'likely': 0, 'unlikely': 0}
            
            for idx, row in raw_with_predictions.iterrows():
                pred_value = row.get('has_booked_prediction')
                confidence = row.get('prediction_confidence', 0.5)
                project_id = row.get('projectid')
                
                if not pd.isna(pred_value) and not pd.isna(project_id):
                    predictions_list.append({
                        'projectid': int(project_id),
                        'row_index': int(idx), # type: ignore
                        'prediction': float(pred_value),
                        'confidence': float(confidence)
                    })
                    
                    # Count predictions
                    if pred_value >= 0.5:
                        prediction_counts['likely'] += 1
                    else:
                        prediction_counts['unlikely'] += 1
            
            if DEBUG_PRINTS:
                print(f"[PREDICT] Generated {len(predictions_list)} predictions for frontend")
                print(f"[PREDICT] Prediction counts: {prediction_counts}")
            
            # 10. Return the raw dataset with predictions
            complete_dataset = sanitize_records(raw_with_predictions.to_dict(orient='records'))
            
            response_data = {
                'predictions': predictions_list,
                'complete_dataset': complete_dataset,
                'raw_preview': sanitize_records(raw_df.head(5).to_dict(orient='records')),
                'processed_preview': sanitize_records(processed_df.head(5).to_dict(orient='records')),
                'prediction_counts': prediction_counts,
                'processed_filename': processed_filename,
                'prediction_results_filename': prediction_results_filename,
                'prediction_results_path': str(prediction_results_path),
                'summary': {
                    'total_rows': len(raw_df),
                    'projects_processed': len(processed_df['projectid'].unique()) if 'projectid' in processed_df.columns else 0,
                    'predictions_generated': len(predictions_list),
                    'prediction_results_saved': True
                }
            }
            
            return jsonify(response_data)
            
        except Exception as e:
            if DEBUG_PRINTS:
                print(f"[PREDICT] Prediction error: {e}")
                import traceback
                traceback.print_exc()
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

@app.route('/api/explainability/<filename>/<int:project_id>', methods=['GET'])
def get_explainability_analysis(filename, project_id):
    """Get explainability analysis for a specific project from prediction results."""
    if DEBUG_PRINTS:
        print(f"[EXPLAIN] Explainability analysis requested for project {project_id} in file {filename}")
    
    try:
        # Look for the prediction results file
        prediction_results_dir = UPLOADS_DIR / "prediction_results"
        if not filename.endswith('_prediction_results.csv'):
            filename = filename.replace('.csv', '_prediction_results.csv')
        
        file_path = prediction_results_dir / filename
        
        if not file_path.exists():
            return jsonify({'error': f'Prediction results file {filename} not found'}), 404
        
        # Read the prediction results
        df = pd.read_csv(file_path, low_memory=False)
        
        if DEBUG_PRINTS:
            print(f"[EXPLAIN] Loaded {len(df)} rows for analysis")
        
        # Filter data for the specific project
        project_data = df[df['projectid'] == project_id].copy()
        
        if len(project_data) == 0:
            return jsonify({'error': f'No data found for project {project_id}'}), 404
        
        # Check if we have predictions
        if 'has_booked_prediction' not in project_data.columns:
            return jsonify({'error': 'No prediction data available'}), 400
        
        # Analyze features by class (0 and 1)
        feature_analysis = analyze_features_by_class(project_data, project_id)
        
        # Find ideal customers for both classes
        ideal_customers = find_ideal_customers_by_class(project_data, feature_analysis, project_id)
        
        # Calculate class distribution
        class_0_count = len(project_data[project_data['has_booked_prediction'] < 0.5])
        class_1_count = len(project_data[project_data['has_booked_prediction'] >= 0.5])
        
        # Prepare response
        response_data = {
            'project_id': project_id,
            'total_samples': len(project_data),
            'class_distribution': {
                'class_0': {
                    'count': class_0_count,
                    'percentage': round(class_0_count / len(project_data) * 100, 2) if len(project_data) > 0 else 0,
                    'label': 'Not Potential Customers'
                },
                'class_1': {
                    'count': class_1_count,
                    'percentage': round(class_1_count / len(project_data) * 100, 2) if len(project_data) > 0 else 0,
                    'label': 'Potential Customers'
                }
            },
            'feature_analysis': feature_analysis,
            'ideal_customers': ideal_customers,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        if DEBUG_PRINTS:
            print(f"[EXPLAIN] Error in explainability analysis: {e}")
            import traceback
            traceback.print_exc()
        return jsonify({'error': f'Error in explainability analysis: {str(e)}'}), 500

def analyze_features_by_class(project_data, project_id):
    """Analyze top 5 most frequent values for each feature per class (0 and 1)."""
    try:
        # Exclude system columns AND ProjectID_Detail.xlsx columns
        exclude_cols = {
            'has_booked_prediction', 'prediction_confidence', 'customerid', 
            'projectid', 'questiondate', 'questiontime', 'fillindate', 
            'saledate', 'bookingdate', 'has_booked',
            # Exclude columns that come from ProjectID_Detail.xlsx merge
            'Project Brand', 'Project Type', 'Location'
        }
        
        # Dynamically detect and exclude any columns that came from ProjectID_Detail.xlsx
        # These typically have title case and are company-related
        for col in project_data.columns:
            if col.startswith('Project ') or col in ['Location', 'Brand', 'Type']:
                exclude_cols.add(col)
        
        if DEBUG_PRINTS:
            print(f"[ANALYZE] Excluding columns: {exclude_cols}")
            print(f"[ANALYZE] Available columns for analysis: {[c for c in project_data.columns if c not in exclude_cols]}")
        
        # Separate data by prediction class
        class_0_data = project_data[project_data['has_booked_prediction'] < 0.5].copy()
        class_1_data = project_data[project_data['has_booked_prediction'] >= 0.5].copy()
        
        feature_analysis = {
            'class_0': {},  # Not potential customers
            'class_1': {}   # Potential customers
        }
        
        for class_label, data in [('class_0', class_0_data), ('class_1', class_1_data)]:
            if len(data) == 0:
                continue
                
            # Process ALL columns except exclude_cols
            for column in data.columns:
                if column in exclude_cols:
                    continue
                
                # Skip columns with too many unique values
                unique_count = data[column].nunique()
                if unique_count > 100:
                    continue
                
                # Skip columns with only one unique value
                if unique_count < 2:
                    continue
                
                # Get top 5 most frequent values
                value_counts = data[column].value_counts().head(5)
                
                if len(value_counts) > 0:
                    feature_analysis[class_label][column] = {
                        'top_values': [
                            {
                                'value': str(value),
                                'count': int(count),
                                'percentage': round(count / len(data) * 100, 1),
                                'is_top_frequent': idx == 0  # Highlight top 1
                            }
                            for idx, (value, count) in enumerate(value_counts.items())
                        ],
                        'total_unique': int(unique_count),
                        'sample_size': len(data)
                    }
        
        return feature_analysis
        
    except Exception as e:
        print(f"Error analyzing features by class: {e}")
        return {'class_0': {}, 'class_1': {}}

def find_ideal_customers_by_class(project_data, feature_analysis, project_id):
    """Find ideal customers for ONLY class_1 (potential customers) based on top frequent features."""
    try:
        ideal_customers = {}
        
        # Only process class_1 (potential customers)
        class_1_data = project_data[project_data['has_booked_prediction'] >= 0.5].copy()
        
        # Only analyze class_1
        if len(class_1_data) > 0 and 'class_1' in feature_analysis and feature_analysis['class_1']:
            customer_scores = []
            
            for idx, row in class_1_data.iterrows():
                score = 0
                matched_features = []
                total_features = 0
                
                for feature_name, feature_data in feature_analysis['class_1'].items():
                    if feature_name in row:
                        total_features += 1
                        top_value = feature_data['top_values'][0]['value']
                        
                        if str(row[feature_name]) == top_value:
                            score += 1
                            matched_features.append({
                                'feature': feature_name,
                                'value': str(row[feature_name]),
                                'frequency_rank': 1,
                                'percentage': feature_data['top_values'][0]['percentage']
                            })
                        else:
                            # Check if it matches any of the top 5 values
                            for i, val_data in enumerate(feature_data['top_values']):
                                if str(row[feature_name]) == val_data['value']:
                                    score += (5 - i) / 5  # Weighted score
                                    matched_features.append({
                                        'feature': feature_name,
                                        'value': str(row[feature_name]),
                                        'frequency_rank': i + 1,
                                        'percentage': val_data['percentage']
                                    })
                                    break
                
                if total_features > 0:
                    match_percentage = round(score / total_features * 100, 1)
                    
                    customer_scores.append({
                        'customer_index': int(idx),
                        'customer_id': str(row.get('customerid', 'Unknown')),
                        'match_score': score,
                        'match_percentage': match_percentage,
                        'matched_features': matched_features,
                        'total_features_checked': total_features,
                        'prediction_confidence': float(row.get('prediction_confidence', 0.5)),
                        'prediction_value': float(row.get('has_booked_prediction', 0.0)),
                        'customer_data': {k: str(v) for k, v in row.items() if k not in {'has_booked_prediction', 'prediction_confidence'}}
                    })
            
            # Sort by match percentage, then by prediction confidence
            customer_scores.sort(
                key=lambda x: (x['match_percentage'], x['prediction_confidence']),
                reverse=True
            )
            
            # Return the most ideal customer for class_1 only
            ideal_customers['class_1'] = customer_scores[0] if customer_scores else None
        else:
            ideal_customers['class_1'] = None
        
        # Don't include class_0 ideal customer
        return ideal_customers
        
    except Exception as e:
        print(f"Error finding ideal customers: {e}")
        return {'class_1': None}

# Add endpoint to get available projects for explainability
@app.route('/api/explainability/<filename>/projects', methods=['GET'])
def get_explainable_projects(filename):
    """Get list of projects available for explainability analysis."""
    try:
        # Look for the prediction results file
        prediction_results_dir = UPLOADS_DIR / "prediction_results"
        if not filename.endswith('_prediction_results.csv'):
            filename = filename.replace('.csv', '_prediction_results.csv')
        
        file_path = prediction_results_dir / filename
        
        if not file_path.exists():
            return jsonify({'error': f'Prediction results file {filename} not found'}), 404
        
        # Read the prediction results
        df = pd.read_csv(file_path, low_memory=False)
        
        # Get unique project IDs with their success counts
        if 'has_booked_prediction' in df.columns:
            project_stats = []
            
            for project_id in sorted(df['projectid'].unique()):
                project_data = df[df['projectid'] == project_id]
                success_cases = project_data[project_data['has_booked_prediction'] >= 0.5]
                
                project_stats.append({
                    'project_id': int(project_id),
                    'total_samples': len(project_data),
                    'success_cases': len(success_cases),
                    'success_rate': round(len(success_cases) / len(project_data) * 100, 1) if len(project_data) > 0 else 0
                })
            
            return jsonify({
                'available_projects': project_stats,
                'total_projects': len(project_stats)
            })
        else:
            return jsonify({'error': 'No prediction data available'}), 400
            
    except Exception as e:
        if DEBUG_PRINTS:
            print(f"Error getting explainable projects: {e}")
        return jsonify({'error': f'Error getting projects: {str(e)}'}), 500

@app.route('/api/prediction_results/<filename>', methods=['GET'])
def get_prediction_results(filename):
    """Load prediction results from the saved file."""
    if DEBUG_PRINTS:
        print(f"[RESULTS] Loading prediction results from: {filename}")
    
    try:
        # Look for the file in prediction_results directory
        prediction_results_dir = UPLOADS_DIR / "prediction_results"
        file_path = prediction_results_dir / filename
        
        if not file_path.exists():
            return jsonify({'error': f'Prediction results file {filename} not found'}), 404
        
        # Read the prediction results file
        if filename.endswith('.csv'):
            df = pd.read_csv(file_path, low_memory=False)
        else:
            df = pd.read_excel(file_path)
        
        if DEBUG_PRINTS:
            print(f"[RESULTS] Loaded {len(df)} rows with {len(df.columns)} columns")
            print(f"[RESULTS] Columns: {list(df.columns)}")
        
        # Convert to records and sanitize for JSON
        complete_dataset = sanitize_records(df.to_dict(orient='records'))
        
        # Extract unique projects
        projects = []
        if 'projectid' in df.columns:
            projects = sorted(df['projectid'].unique().tolist())
        
        # Calculate statistics
        stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'unique_projects': len(projects),
            'has_predictions': 'has_booked_prediction' in df.columns,
            'predictions_count': len(df[df['has_booked_prediction'].notna()]) if 'has_booked_prediction' in df.columns else 0
        }
        
        if 'has_booked_prediction' in df.columns:
            # Calculate prediction distribution
            prediction_counts = {
                'likely': len(df[df['has_booked_prediction'] >= 0.5]),
                'unlikely': len(df[df['has_booked_prediction'] < 0.5])
            }
            stats['prediction_distribution'] = prediction_counts
        
        response_data = {
            'success': True,
            'complete_dataset': complete_dataset,
            'available_projects': projects,
            'stats': stats,
            'filename': filename,
            'columns': list(df.columns)
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        if DEBUG_PRINTS:
            print(f"[RESULTS] Error loading prediction results: {e}")
            import traceback
            traceback.print_exc()
        return jsonify({'error': f'Error loading prediction results: {str(e)}'}), 500
        

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