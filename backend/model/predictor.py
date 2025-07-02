import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any
import joblib
import os
from pathlib import Path

# Add SHAP for model interpretation
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")

# Debug configuration - set to False to disable debug prints
DEBUG_PRINTS = False

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
    
    def add_prediction_confidence(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add prediction confidence scores to the DataFrame.
        
        Args:
            data (pd.DataFrame): DataFrame with predictions
            
        Returns:
            pd.DataFrame: DataFrame with added confidence scores
        """
        if 'has_booked_prediction' not in data.columns:
            raise ValueError("DataFrame must contain 'has_booked_prediction' column")
        
        data = data.copy()
        confidences = np.full(len(data), 0.5)  # Default confidence
        
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
                # Get the data for this project
                if features:
                    X = data.loc[mask, features]
                else:
                    X = data.loc[mask].drop(columns=['has_booked_prediction', 'prediction_confidence'], errors='ignore')
                
                if isinstance(X, pd.Series):
                    X = X.to_frame().T
                
                # Try to get prediction probabilities for confidence
                if hasattr(model, 'predict_proba'):
                    try:
                        probas = model.predict_proba(X)
                        # Use the maximum probability as confidence
                        confidences[mask] = np.max(probas, axis=1)
                    except Exception:
                        # If predict_proba fails, use decision function if available
                        if hasattr(model, 'decision_function'):
                            try:
                                decisions = model.decision_function(X)
                                # Convert decision scores to confidence (0.5 to 1.0 range)
                                confidences[mask] = 0.5 + 0.5 * (np.abs(decisions) / (1 + np.abs(decisions)))
                            except Exception:
                                confidences[mask] = 0.75  # Default confidence
                        else:
                            confidences[mask] = 0.75  # Default confidence
                else:
                    confidences[mask] = 0.75  # Default confidence
            except Exception as e:
                if DEBUG_PRINTS:
                    print(f"Confidence calculation failed for project {pid}: {e}")
                confidences[mask] = 0.5  # Low confidence for errors
        
        data['prediction_confidence'] = confidences
        return data

    def calculate_feature_importance(self, data: pd.DataFrame, project_id: int) -> Dict[str, Any]:
        """
        Calculate feature importance using SHAP values for a specific project.
        
        Args:
            data (pd.DataFrame): Input data with features
            project_id (int): Project ID for model selection
            
        Returns:
            Dict[str, Any]: Dictionary with feature importance information
        """
        if not SHAP_AVAILABLE:
            return {"error": "SHAP not available. Install with: pip install shap"}
            
        try:
            # Check if model exists for this project
            if project_id not in self.models:
                return {"error": f"No model available for project ID {project_id}"}
                
            # Get model and features for this project
            model, features = self.models[project_id]
            
            # Extract features needed for this model
            if features:
                X = data.loc[data['projectid'] == project_id, features]
            else:
                X = data.loc[data['projectid'] == project_id].drop(
                    columns=['has_booked_prediction', 'prediction_confidence', 'projectid'], errors='ignore'
                )
                features = X.columns.tolist()
                
            if len(X) == 0:
                return {"error": f"No data available for project ID {project_id}"}
                
            # Handle series to dataframe conversion
            if isinstance(X, pd.Series):
                X = X.to_frame().T
                
            # Calculate SHAP values based on model type
            model_type = type(model).__name__
            
            # Initialize SHAP explainer based on model type
            if hasattr(model, 'predict_proba'):
                # For classifiers with predict_proba
                if model_type in ['RandomForestClassifier', 'GradientBoostingClassifier', 'XGBClassifier']:
                    explainer = shap.TreeExplainer(model)
                elif model_type in ['LogisticRegression', 'SGDClassifier', 'LinearSVC']:
                    explainer = shap.LinearExplainer(model, X)
                else:
                    # Fallback to KernelExplainer (slower but works with any model)
                    explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X, 50))
            else:
                # For regressors without predict_proba
                if model_type in ['RandomForestRegressor', 'GradientBoostingRegressor', 'XGBRegressor']:
                    explainer = shap.TreeExplainer(model)
                elif model_type in ['LinearRegression', 'Lasso', 'Ridge', 'ElasticNet']:
                    explainer = shap.LinearExplainer(model, X)
                else:
                    explainer = shap.KernelExplainer(model.predict, shap.sample(X, 50))
            
            # Calculate SHAP values
            # Use a sample of data if there are too many rows
            if len(X) > 100:
                X_sample = X.sample(100, random_state=42)
            else:
                X_sample = X
                
            shap_values = explainer.shap_values(X_sample)
            
            # For classifiers, shap_values might be a list of arrays (one per class)
            if isinstance(shap_values, list):
                # Take the class 1 (positive class) SHAP values for binary classification
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                
            # Calculate mean absolute SHAP values for each feature
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            feature_importance = dict(zip(features, mean_abs_shap))
            
            # Sort features by importance
            sorted_importance = {k: v for k, v in sorted(feature_importance.items(), 
                                                       key=lambda item: item[1], reverse=True)}
            
            # Normalize to percentages
            total_importance = sum(sorted_importance.values())
            importance_pct = {k: (v / total_importance) * 100 for k, v in sorted_importance.items()}
            
            # Generate data for visualizations
            shap_data = {
                'feature_names': list(sorted_importance.keys()),
                'importance_values': list(sorted_importance.values()),
                'importance_pct': list(importance_pct.values()),
                'raw_shap_values': shap_values.tolist() if isinstance(shap_values, np.ndarray) else None,
                'raw_data_sample': X_sample.to_dict(orient='records'),
                'model_type': model_type,
                'project_id': project_id
            }
            
            return {
                'status': 'success',
                'feature_importance': sorted_importance,
                'importance_pct': importance_pct,
                'shap_data': shap_data,
                'total_features': len(features)
            }
                
        except Exception as e:
            if DEBUG_PRINTS:
                print(f"Feature importance calculation failed: {e}")
            import traceback
            return {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def analyze_all_projects(self, data: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        """
        Generate feature importance analysis for all available projects in the data.
        
        Args:
            data (pd.DataFrame): Input data with features
            
        Returns:
            Dict[int, Dict[str, Any]]: Dictionary with feature importance per project
        """
        analysis_results = {}
        
        # Get unique project IDs in the data
        project_ids = data['projectid'].unique()
        
        for project_id in project_ids:
            try:
                pid = int(project_id)
                # Skip if model not available
                if pid not in self.models:
                    analysis_results[pid] = {"error": f"No model available for project {pid}"}
                    continue
                    
                # Calculate feature importance for this project
                result = self.calculate_feature_importance(data, pid)
                analysis_results[pid] = result
                
            except Exception as e:
                if DEBUG_PRINTS:
                    print(f"Analysis failed for project {project_id}: {e}")
                analysis_results[project_id] = {"error": str(e)}
                
        return analysis_results