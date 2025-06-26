from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import tempfile
import json
from werkzeug.utils import secure_filename
from model.predictor import MLPredictor
import io

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
MAX_ROWS = 100000
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize the ML predictor
predictor = MLPredictor()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_file(file_path):
    """Read CSV or Excel file and return DataFrame"""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    else:
        return pd.read_excel(file_path)

# Utility function to convert DataFrame to records with native Python types
def df_to_records_native(df):
    return json.loads(df.to_json(orient='records'))

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload and return preview"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not file or not file.filename:
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(str(file.filename)):
            return jsonify({'error': 'Invalid file type. Please upload CSV or Excel files.'}), 400
        
        # Always save as CSV for faster preprocessing
        ext = file.filename.rsplit('.', 1)[-1].lower()
        filename = secure_filename(str(file.filename))
        csv_filename = f"{filename.rsplit('.', 1)[0]}.csv"
        file_path = os.path.join(UPLOAD_FOLDER, csv_filename)
        
        # Read the uploaded file and convert to CSV
        if ext == 'csv':
            df = pd.read_csv(file.stream)
        else:
            df = pd.read_excel(file.stream, engine='openpyxl' if ext == 'xlsx' else 'xlrd')
        
        # Save as CSV for faster preprocessing
        df.to_csv(file_path, index=False)
        
        # Check file size
        if len(df) > MAX_ROWS:
            os.remove(file_path)
            return jsonify({'error': f'File too large. Maximum {MAX_ROWS:,} rows allowed.'}), 400
        
        # Check if projectid column exists
        if 'projectid' not in df.columns:
            os.remove(file_path)
            return jsonify({'error': 'File must contain a "projectid" column.'}), 400
        
        # Get preview (first 5 and last 5 rows)
        preview_rows = 5
        if len(df) <= 10:
            preview_data = df_to_records_native(df)
        else:
            first_rows = df.head(preview_rows)
            last_rows = df.tail(preview_rows)
            preview_data = df_to_records_native(pd.concat([first_rows, last_rows]))
        
        # Get unique project IDs
        unique_projects = df['projectid'].unique().tolist()
        
        # Check which models are available
        available_models = []
        missing_models = []
        for project_id in unique_projects:
            model_path = os.path.join('model', f'project_{int(project_id)}_model.pkl')
            if os.path.exists(model_path):
                available_models.append(int(project_id))
            else:
                missing_models.append(int(project_id))
        
        return jsonify({
            'filename': csv_filename,
            'total_rows': len(df),
            'preview_data': preview_data,
            'columns': df.columns.tolist(),
            'unique_projects': unique_projects,
            'available_models': available_models,
            'missing_models': missing_models,
            'preview_rows': preview_rows
        })
    
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Generate predictions for uploaded file"""
    try:
        data = request.json
        filename = data.get('filename') if data else None
        
        if not filename:
            return jsonify({'error': 'Filename not provided'}), 400
        
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        df = read_file(file_path)
        
        # Batch prediction, progress logging for large files
        df_with_predictions = predictor.predict(df, log_progress=True)
        
        # Ensure has_booked_prediction is last column
        if 'has_booked_prediction' not in df_with_predictions.columns:
            return jsonify({'error': 'Prediction column missing after prediction step.'}), 500
        cols = [c for c in df_with_predictions.columns if c != 'has_booked_prediction'] + ['has_booked_prediction']
        df_with_predictions = df_with_predictions.reindex(columns=cols)
        
        # Now generate preview data
        preview_rows = 5
        if len(df_with_predictions) <= 10:
            preview_data = df_to_records_native(df_with_predictions)
        else:
            first_rows = df_with_predictions.head(preview_rows)
            last_rows = df_with_predictions.tail(preview_rows)
            preview_data = df_to_records_native(pd.concat([first_rows, last_rows]))
        
        # Save results (with prediction column)
        results_filename = f"results_{filename}"
        results_path = os.path.join(UPLOAD_FOLDER, results_filename)
        
        # Always save results as CSV for consistency
        df_with_predictions.to_csv(results_path, index=False)
        
        # Calculate prediction statistics
        prediction_stats = {
            'total_predictions': int(len(df_with_predictions)),
            'successful_predictions': int(pd.notna(df_with_predictions['has_booked_prediction']).sum()),
            'failed_predictions': int(pd.isna(df_with_predictions['has_booked_prediction']).sum()),
            'prediction_rate': float(round(pd.notna(df_with_predictions['has_booked_prediction']).sum() / len(df_with_predictions) * 100, 2))
        }
        
        return jsonify({
            'results_filename': results_filename,
            'preview_data': preview_data,
            'prediction_stats': prediction_stats,
            'columns': df_with_predictions.columns.tolist()
        })
    
    except Exception as e:
        return jsonify({'error': f'Error generating predictions: {str(e)}'}), 500

@app.route('/api/export/<format>/<filename>', methods=['GET'])
def export_file(format, filename):
    """Export results in specified format"""
    try:
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        df = read_file(file_path)
        
        if format == 'json':
            # Return JSON data
            return jsonify(df_to_records_native(df))
        
        elif format == 'csv':
            # Return CSV file
            output = io.StringIO()
            df.to_csv(output, index=False)
            output.seek(0)
            return send_file(
                io.BytesIO(output.getvalue().encode('utf-8')),
                mimetype='text/csv',
                as_attachment=True,
                download_name=f"{filename.rsplit('.', 1)[0]}.csv"
            )
        
        elif format == 'xlsx':
            # Return Excel file
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            output.seek(0)
            return send_file(
                output,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                as_attachment=True,
                download_name=f"{filename.rsplit('.', 1)[0]}.xlsx"
            )
        
        else:
            return jsonify({'error': 'Invalid export format'}), 400
    
    except Exception as e:
        return jsonify({'error': f'Error exporting file: {str(e)}'}), 500

@app.route('/api/cleanup', methods=['POST'])
def cleanup():
    """Clean up uploaded files"""
    try:
        data = request.json
        filename = data.get('filename')
        
        if filename:
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
            
            # Also remove results file if it exists
            results_filename = f"results_{filename}"
            results_path = os.path.join(UPLOAD_FOLDER, results_filename)
            if os.path.exists(results_path):
                os.remove(results_path)
        
        return jsonify({'message': 'Cleanup completed'})
    
    except Exception as e:
        return jsonify({'error': f'Error during cleanup: {str(e)}'}), 500

@app.route('/api/models', methods=['GET'])
def get_available_models():
    """Get list of available models"""
    try:
        model_files = [f for f in os.listdir('model') if f.endswith('.pkl')]
        project_ids = []
        
        for model_file in model_files:
            try:
                project_id = model_file.replace('project_', '').replace('_model.pkl', '')
                if project_id is not None:
                    project_ids.append(int(project_id))
            except:
                continue
        
        return jsonify({
            'available_models': sorted(project_ids),
            'total_models': len(project_ids)
        })
    
    except Exception as e:
        return jsonify({'error': f'Error getting models: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 