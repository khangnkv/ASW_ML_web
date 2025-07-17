import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any
import joblib
import os
import sys
import warnings
from pathlib import Path
import traceback
from backend.model.explainability_utils import extract_feature_importance, calculate_conditional_analysis

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Debug configuration
DEBUG_PRINTS = False

class MLPredictor:
    def __init__(self):
        """Initialize the ML predictor."""
        self.model_dir = Path(__file__).parent
        self.models = {}
        self._preload_models()

    def _preload_models(self):
        """Preload all models into memory at startup."""
        if not self.model_dir.exists():
            print(f"Warning: Model directory {self.model_dir} does not exist")
            return
            
        for fname in os.listdir(self.model_dir):
            if fname.startswith("project_") and fname.endswith("_model.pkl"):
                try:
                    project_id = int(fname.replace("project_", "").replace("_model.pkl", ""))
                    model_data = joblib.load(os.path.join(self.model_dir, fname))
                    model = model_data.get("model")
                    print(f"Loaded model for project {project_id}: {type(model)}")
                    features = model_data.get("features", [])
                    self.models[project_id] = (model, features)
                except Exception as e:
                    print(f"Failed to load model {fname}: {e}")

    def predict(self, data: pd.DataFrame, log_progress: bool = True) -> pd.DataFrame:
        """Generate predictions for all projects in the data."""
        if 'projectid' not in data.columns:
            raise ValueError("Input DataFrame must contain a 'projectid' column.")
        data = data.copy()
        predictions = np.full(len(data), np.nan)
        missing_models = set()
        n_rows = len(data)
        unique_projects = data['projectid'].unique()
        
        for i, project_id in enumerate(unique_projects):
            try:
                pid = int(project_id)
            except Exception:
                if DEBUG_PRINTS:
                    print(f"Invalid project_id: {project_id}")
                missing_models.add(project_id)
                continue
                
            if log_progress and n_rows > 60000:
                if DEBUG_PRINTS:
                    print(f"[Predict] Processing project_id={pid} ({i+1}/{len(unique_projects)})")
                    
            mask = (data['projectid'] == project_id) | (data['projectid'] == pid)
            
            if pid not in self.models:
                if DEBUG_PRINTS:
                    print(f"Model for project_id {pid} not found. Available: {list(self.models.keys())}")
                missing_models.add(pid)
                continue
                
            model, features = self.models[pid]
            
            try:
                if features:
                    X = data.loc[mask, features]
                else:
                    X = data.loc[mask].drop(columns=['has_booked_prediction'], errors='ignore')
                    
                if isinstance(X, pd.Series):
                    X = X.to_frame().T
                    
                if log_progress:
                    if DEBUG_PRINTS:
                        print(f"Predicting for project_id={pid}, features={list(X.columns)}")
                        
                preds = model.predict(X)
                predictions[mask] = preds
                
            except Exception as e:
                if DEBUG_PRINTS:
                    print(f"Prediction failed for project {pid}: {e}")
                predictions[mask] = np.nan
        
        data['has_booked_prediction'] = predictions
        
        # Ensure prediction columns are at the end
        prediction_cols = ['has_booked_prediction', 'prediction_confidence']
        other_cols = [c for c in data.columns if c not in prediction_cols]
        existing_pred_cols = [c for c in prediction_cols if c in data.columns]
        cols = other_cols + existing_pred_cols
        data = data[cols]
        
        if missing_models:
            if DEBUG_PRINTS:
                print(f"Warning: Missing models for project IDs: {sorted(missing_models)}")
                
        return data

    def predict_single_project(self, data, project_id):
        """Generate predictions for a single project ID"""
        if project_id not in self.models:
            raise ValueError(f"No model available for project {project_id}")
        
        model, expected_features = self.models[project_id]
        
        # Ensure data has the expected features
        missing_features = set(expected_features) - set(data.columns)
        if missing_features:
            print(f"Warning: Missing features for project {project_id}: {missing_features}")
            for feature in missing_features:
                data[feature] = 0
        
        # Select only the expected features in the correct order
        X = data[expected_features]
        
        # Generate predictions
        predictions = model.predict(X)
        return predictions

    def add_prediction_confidence(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add prediction confidence scores to the DataFrame."""
        if 'has_booked_prediction' not in data.columns:
            raise ValueError("DataFrame must contain 'has_booked_prediction' column")
        
        data = data.copy()
        confidences = np.full(len(data), 0.5)
        
        unique_projects = data['projectid'].unique()
        for project_id in unique_projects:
            try:
                pid = int(project_id)
            except Exception:
                continue
                
            if pid not in self.models:
                continue
                
            model, features = self.models[pid]
            mask = (data['projectid'] == project_id) | (data['projectid'] == pid)
            
            try:
                if features:
                    X = data.loc[mask, features]
                else:
                    X = data.loc[mask].drop(columns=['has_booked_prediction', 'prediction_confidence'], errors='ignore')
                
                if isinstance(X, pd.Series):
                    X = X.to_frame().T
                
                if hasattr(model, 'predict_proba'):
                    try:
                        probas = model.predict_proba(X)
                        confidences[mask] = np.max(probas, axis=1)
                    except Exception:
                        confidences[mask] = 0.75
                else:
                    confidences[mask] = 0.75
                    
            except Exception as e:
                if DEBUG_PRINTS:
                    print(f"Confidence calculation failed for project {pid}: {e}")
                confidences[mask] = 0.5
        
        data['prediction_confidence'] = confidences
        return data

    def explain_project(self, data: pd.DataFrame, project_id: int) -> Dict[str, Any]:
        """
        Generate explainability analysis for a specific project using feature importance 
        and conditional analysis.
        
        Args:
            data: DataFrame with predictions (should contain prediction column)
            project_id: Project ID to analyze
        
        Returns:
            Dictionary with explainability results
        """
        try:
            # Validate project_id
            if project_id not in self.models:
                available_projects = list(self.models.keys())
                return {
                    'success': False,
                    'error': f"No model available for project {project_id}. Available projects: {available_projects}",
                    'project_id': project_id
                }
            
            # Convert project_id types for consistent filtering
            data_copy = data.copy()
            data_copy['projectid'] = data_copy['projectid'].astype(str)
            project_id_str = str(project_id)
            
            # Filter data for this project
            project_mask = data_copy["projectid"] == project_id_str
            project_data = data_copy.loc[project_mask].copy()
            
            if project_data.empty:
                return {
                    'success': False,
                    'error': f"No data rows found for project {project_id}. Available projects in data: {data_copy['projectid'].unique().tolist()}",
                    'project_id': project_id
                }
            
            if DEBUG_PRINTS:
                print(f"Explaining project {project_id} with {len(project_data)} rows")
                print(f"Available columns: {list(project_data.columns)}")
            
            explanation_result = {
                'success': True,
                'project_id': project_id,
                'sample_size': len(project_data)
            }
            
            # 1. Feature Importance Analysis
            model_data = self.models[project_id]
            feature_importance_result = extract_feature_importance(model_data, project_id)
            explanation_result['feature_importance'] = feature_importance_result
            
            # 2. Conditional Dataset Analysis
            conditional_analysis_result = calculate_conditional_analysis(project_data, project_id)
            explanation_result['conditional_analysis'] = conditional_analysis_result
            
            return explanation_result
            
        except Exception as e:
            if DEBUG_PRINTS:
                print(f"Error in explain_project for project {project_id}: {e}")
                import traceback
                traceback.print_exc()
            
            return {
                'success': False,
                'error': str(e),
                'project_id': project_id
            }