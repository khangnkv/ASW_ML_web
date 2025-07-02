import numpy as np
import pandas as pd
from typing import Union, List
import joblib
import os
from pathlib import Path

class MLPredictor:
    def __init__(self):
        """
        Initialize the ML predictor.
        
        Args:
            model_dir (str): Directory where project models are stored.
        """
        # Fix the model directory path for Docker
        self.model_dir = Path(__file__).parent  # This will be /app/backend/model in Docker
        # Alternative: you can also use an absolute path
        # self.model_dir = Path('/app/backend/model')
        
        self.models = {}  # <-- Ensure models dict is always initialized
        self._preload_models()  # type: ignore

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
        """
        For each project, use batch prediction and log progress if requested.
        Adds a 'has_booked_prediction' column to the DataFrame as the last column.
        """
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
                print(f"Invalid project_id: {project_id}")
                missing_models.add(project_id)
                continue
            if log_progress and n_rows > 60000:
                print(f"[Predict] Processing project_id={pid} ({i+1}/{len(unique_projects)})")
            mask = (data['projectid'] == project_id) | (data['projectid'] == pid)
            if pid not in self.models:
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
                    print(f"Predicting for project_id={pid}, features={list(X.columns)}")
                preds = model.predict(X)
                predictions[mask] = preds
            except Exception as e:
                print(f"Prediction failed for project {pid}: {e}")
                predictions[mask] = np.nan
        data['has_booked_prediction'] = predictions
        cols = [c for c in data.columns if c != 'has_booked_prediction'] + ['has_booked_prediction']
        data = pd.DataFrame(data, columns=cols)
        if missing_models:
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
            # Add missing features with default values
            for feature in missing_features:
                data[feature] = 0
        
        # Select only the expected features in the correct order
        X = data[expected_features]
        
        # Generate predictions
        predictions = model.predict(X)
        return predictions

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities if the model supports it.
        
        Args:
            data (pd.DataFrame): Processed input data
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        if not hasattr(self.model, 'predict_proba'):
            raise NotImplementedError("Model does not support probability predictions")
            
        try:
            probabilities = self.model.predict_proba(data)
            return probabilities
        except Exception as e:
            raise Exception(f"Error getting prediction probabilities: {str(e)}")