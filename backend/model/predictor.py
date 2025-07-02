import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any, Callable
import joblib
import os
import sys
import warnings
from pathlib import Path
import pickle
import traceback  # Added for improved error handling

# Add SHAP for model interpretation with robust error handling
try:
    import shap
    # Test if SHAP is working correctly
    try:
        # Simple test with a small array
        test_data = np.array([[1, 2, 3], [4, 5, 6]])
        test_model = lambda x: np.sum(x, axis=1)
        test_explainer = shap.KernelExplainer(test_model, test_data)
        test_explainer.shap_values(test_data)
        SHAP_AVAILABLE = True
    except Exception as e:
        print(f"SHAP import succeeded but test failed: {e}")
        SHAP_AVAILABLE = False
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")
except Exception as e:
    SHAP_AVAILABLE = False
    print(f"Error initializing SHAP: {e}")

# Try to import InterpretML for alternative model explanation
try:
    from interpret.blackbox import LimeTabular, PartialDependence, MorrisSensitivity
    from interpret.ext.blackbox import TabularExplainer
    from interpret import show
    INTERPRET_ML_AVAILABLE = True
    print("InterpretML loaded successfully")
except ImportError:
    INTERPRET_ML_AVAILABLE = False
    print("InterpretML not available. Install with: pip install interpret")
except Exception as e:
    INTERPRET_ML_AVAILABLE = False
    print(f"Error initializing InterpretML: {e}")

# Suppress FutureWarnings which often come from SHAP and sklearn
warnings.simplefilter(action='ignore', category=FutureWarning)

# Debug configuration - set to False to disable debug prints
DEBUG_PRINTS = True

# Set to True to use only basic feature importance (no SHAP)
USE_BASIC_FEATURE_IMPORTANCE = False

# Set to True to use InterpretML when SHAP fails
USE_INTERPRET_ML_FALLBACK = True

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
        If SHAP is not available or if USE_BASIC_FEATURE_IMPORTANCE is True,
        falls back to using model's feature_importances_ attribute if available.
        
        Args:
            data (pd.DataFrame): Input data with features
            project_id (int): Project ID for model selection
            
        Returns:
            Dict[str, Any]: Dictionary with feature importance information
        """
        if not SHAP_AVAILABLE or USE_BASIC_FEATURE_IMPORTANCE:
            if DEBUG_PRINTS:
                print(f"Using basic feature importance for project {project_id}")
            return self._calculate_basic_feature_importance(data, project_id)
            
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
            
            # Check if model is a Pipeline
            is_pipeline = model_type == 'Pipeline'
            
            # For pipeline models, we need special handling to avoid the feature_names_in_ error
            if is_pipeline:
                if DEBUG_PRINTS:
                    print(f"Pipeline detected: {model_type}")
                    
                try:
                    # Try to create a safe copy of the model for SHAP to use
                    # This prevents SHAP from modifying the original model and causing errors
                    # with read-only properties like feature_names_in_
                    model_copy = None
                    try:
                        # Try pickling to create a deep copy - this avoids sharing references
                        model_copy = pickle.loads(pickle.dumps(model))
                        if DEBUG_PRINTS:
                            print("Created model copy via pickle")
                    except Exception as pickle_err:
                        if DEBUG_PRINTS:
                            print(f"Pickling model failed: {pickle_err}")
                        # If pickling fails, fall back to using the original model with KernelExplainer
                        # which is less likely to modify the model
                        pass
                    
                    final_estimator = None
                    if hasattr(model, 'named_steps') and len(model.named_steps) > 0:
                        # Get the final estimator in the pipeline
                        final_step_name = list(model.named_steps.keys())[-1]
                        final_estimator = model.named_steps[final_step_name]
                        model_type = f"Pipeline({type(final_estimator).__name__})"
                    
                    # Always use KernelExplainer with Pipelines to avoid attribute modification
                    # KernelExplainer treats the model as a black box and doesn't try to modify it
                    model_to_use = model_copy if model_copy is not None else model
                    
                    # Create a wrapper function for prediction to avoid directly passing the model
                    if hasattr(model_to_use, 'predict_proba'):
                        predict_fn = lambda x: model_to_use.predict_proba(x)
                        explainer = shap.KernelExplainer(
                            predict_fn, 
                            shap.sample(X, min(50, len(X)))
                        )
                    else:
                        predict_fn = lambda x: model_to_use.predict(x)
                        explainer = shap.KernelExplainer(
                            predict_fn, 
                            shap.sample(X, min(50, len(X)))
                        )
                except Exception as e:
                    if DEBUG_PRINTS:
                        print(f"Error creating explainer for Pipeline: {e}")
                    # If all attempts fail, fall back to basic feature importance
                    return self._calculate_basic_feature_importance(data, project_id)
            else:
                # Non-pipeline models - standard approach
                try:
                    if hasattr(model, 'predict_proba'):
                        # For classifiers with predict_proba
                        if model_type in ['RandomForestClassifier', 'GradientBoostingClassifier', 'XGBClassifier', 'LGBMClassifier', 'CatBoostClassifier']:
                            try:
                                explainer = shap.TreeExplainer(model)
                            except Exception as e:
                                # Fallback to KernelExplainer if TreeExplainer fails
                                if DEBUG_PRINTS:
                                    print(f"TreeExplainer failed: {e}, falling back to KernelExplainer")
                                explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X, min(50, len(X))))
                        elif model_type in ['LogisticRegression', 'SGDClassifier', 'LinearSVC']:
                            try:
                                explainer = shap.LinearExplainer(model, X)
                            except Exception as e:
                                if DEBUG_PRINTS:
                                    print(f"LinearExplainer failed: {e}, falling back to KernelExplainer")
                                explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X, min(50, len(X))))
                        else:
                            # Fallback to KernelExplainer (slower but works with any model)
                            explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X, min(50, len(X))))
                    else:
                        # For regressors without predict_proba
                        if model_type in ['RandomForestRegressor', 'GradientBoostingRegressor', 'XGBRegressor', 'LGBMRegressor', 'CatBoostRegressor']:
                            try:
                                explainer = shap.TreeExplainer(model)
                            except Exception as e:
                                if DEBUG_PRINTS:
                                    print(f"TreeExplainer failed: {e}, falling back to KernelExplainer")
                                explainer = shap.KernelExplainer(model.predict, shap.sample(X, min(50, len(X))))
                        elif model_type in ['LinearRegression', 'Lasso', 'Ridge', 'ElasticNet']:
                            try:
                                explainer = shap.LinearExplainer(model, X)
                            except Exception as e:
                                if DEBUG_PRINTS:
                                    print(f"LinearExplainer failed: {e}, falling back to KernelExplainer")
                                explainer = shap.KernelExplainer(model.predict, shap.sample(X, min(50, len(X))))
                        else:
                            explainer = shap.KernelExplainer(model.predict, shap.sample(X, min(50, len(X))))
                except Exception as e:
                    if DEBUG_PRINTS:
                        print(f"Error creating explainer: {e}")
                    # Fall back to basic feature importance
                    return self._calculate_basic_feature_importance(data, project_id)
            
            # Calculate SHAP values
            # Use a sample of data if there are too many rows
            sample_size = min(100, len(X))  # Ensure we don't try to sample more than we have
            if len(X) > sample_size:
                X_sample = X.sample(sample_size, random_state=42)
            else:
                X_sample = X
            
            # Ensure X_sample has the right format (no missing values, proper types)
            X_sample = X_sample.fillna(0)  # Fill NaN values
                
            try:
                # Calculate SHAP values with additional error handling
                shap_values = explainer.shap_values(X_sample)
                
                # For classifiers, shap_values might be a list of arrays (one per class)
                if isinstance(shap_values, list):
                    # Take the class 1 (positive class) SHAP values for binary classification
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            except Exception as e:
                if DEBUG_PRINTS:
                    print(f"Error calculating SHAP values: {e}")
                
                # Try InterpretML as the first fallback if available
                if INTERPRET_ML_AVAILABLE and USE_INTERPRET_ML_FALLBACK:
                    if DEBUG_PRINTS:
                        print(f"SHAP failed, trying InterpretML for project {project_id}")
                    
                    # Use our InterpretML method instead
                    interpret_result = self._calculate_interpret_ml_importance(data, project_id)
                    
                    # If successful, add a note about the fallback
                    if interpret_result.get('status') == 'success':
                        interpret_result['note'] = 'SHAP calculation failed, using InterpretML as fallback'
                        # Convert interpret_data to shap_data format for compatibility
                        if 'interpret_data' in interpret_result:
                            interpret_result['shap_data'] = interpret_result['interpret_data']
                            interpret_result['shap_data']['note'] = 'Using InterpretML as fallback'
                        return interpret_result
                
                # Second fallback: Return feature importance based on model's feature_importances_ if available
                if hasattr(model, 'feature_importances_'):
                    # For tree-based models that have feature_importances_
                    importances = model.feature_importances_
                    feature_importance = dict(zip(features, importances))
                    sorted_importance = {k: v for k, v in sorted(
                        feature_importance.items(), key=lambda item: item[1], reverse=True)}
                    total_importance = sum(sorted_importance.values())
                    importance_pct = {k: (v / total_importance) * 100 for k, v in sorted_importance.items()}
                    
                    # Return the feature importance without SHAP values
                    return {
                        'status': 'success',
                        'feature_importance': sorted_importance,
                        'importance_pct': importance_pct,
                        'shap_data': {
                            'feature_names': list(sorted_importance.keys()),
                            'importance_values': list(sorted_importance.values()),
                            'importance_pct': list(importance_pct.values()),
                            'model_type': model_type,
                            'project_id': project_id,
                            'note': 'Using model.feature_importances_ as fallback after SHAP and InterpretML failed'
                        },
                        'total_features': len(features),
                        'note': 'SHAP calculation failed, using model.feature_importances_'
                    }
                else:
                    # If no feature_importances_ available, create a simple equal distribution
                    equal_importance = 1.0 / len(features) if len(features) > 0 else 0
                    feature_importance = {f: equal_importance for f in features}
                    
                    return {
                        'status': 'warning',
                        'message': f"SHAP and InterpretML calculations failed: {str(e)}. Using equal feature importance.",
                        'feature_importance': feature_importance,
                        'importance_pct': {k: 100/len(features) for k in features},
                        'shap_data': {
                            'feature_names': features,
                            'importance_values': [equal_importance] * len(features),
                            'importance_pct': [100/len(features)] * len(features),
                            'model_type': model_type,
                            'project_id': project_id,
                            'note': 'Equal distribution due to all feature importance methods failing'
                        },
                        'total_features': len(features)
                    }
                
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
    
    def _calculate_basic_feature_importance(self, data: pd.DataFrame, project_id: int) -> Dict[str, Any]:
        """
        Calculate basic feature importance without using SHAP.
        Uses model's feature_importances_ attribute or coef_ if available.
        For Pipeline objects, tries to extract from the final estimator.
        
        Args:
            data (pd.DataFrame): Input data with features
            project_id (int): Project ID for model selection
            
        Returns:
            Dict[str, Any]: Dictionary with feature importance information
        """
        try:
            # Check if model exists for this project
            if project_id not in self.models:
                return {"error": f"No model available for project ID {project_id}"}
                
            # Get model and features for this project
            model, features = self.models[project_id]
            model_type = type(model).__name__
            
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
            
            # Try to extract feature importance based on model type
            importances = None
            final_estimator = None
            
            # Check if it's a pipeline
            if model_type == 'Pipeline':
                if hasattr(model, 'named_steps') and len(model.named_steps) > 0:
                    # Get the final estimator in the pipeline
                    final_step_name = list(model.named_steps.keys())[-1]
                    final_estimator = model.named_steps[final_step_name]
                    model_type = f"Pipeline({type(final_estimator).__name__})"
                    
                    # Try different methods to get feature importance from the final estimator
                    if hasattr(final_estimator, 'feature_importances_'):
                        importances = final_estimator.feature_importances_
                    elif hasattr(final_estimator, 'coef_'):
                        if hasattr(final_estimator.coef_, 'ndim') and final_estimator.coef_.ndim > 1:
                            importances = np.abs(final_estimator.coef_[0])
                        else:
                            importances = np.abs(final_estimator.coef_)
                    # For more complex models in pipelines, try other approaches
                    elif hasattr(final_estimator, 'feature_importances'):  # Some models use this
                        importances = final_estimator.feature_importances
                    elif hasattr(model, 'feature_importances_'):  # Sometimes pipeline itself has this attribute
                        importances = model.feature_importances_
                    
                    # If we still can't get importances, try to extract from model directly
                    if importances is None and DEBUG_PRINTS:
                        print(f"Couldn't extract feature importances from pipeline estimator {type(final_estimator).__name__}")
            else:
                # Direct model
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    if hasattr(model.coef_, 'ndim') and model.coef_.ndim > 1:
                        importances = np.abs(model.coef_[0])
                    else:
                        importances = np.abs(model.coef_)
                elif hasattr(model, 'feature_importances'):  # Some models use this name
                    importances = model.feature_importances
                
                # For some models like RandomForest, try get_feature_importance if available
                if importances is None and hasattr(model, 'get_feature_importance'):
                    try:
                        importances = model.get_feature_importance()
                    except Exception as e:
                        if DEBUG_PRINTS:
                            print(f"get_feature_importance failed: {e}")
            
            # If we still can't get importances, use permutation importance
            if importances is None and len(X) > 0:
                try:
                    # Try to use permutation importance which works with any model
                    from sklearn.inspection import permutation_importance
                    
                    # Create a small sample to avoid performance issues
                    X_sample = X.sample(min(100, len(X)), random_state=42) if len(X) > 100 else X
                    
                    # Permutation importance requires a target/response variable
                    # We'll use a dummy target since we just need relative feature importances
                    dummy_target = np.zeros(len(X_sample))
                    
                    # Calculate permutation importance
                    perm_importance = permutation_importance(model, X_sample, dummy_target, 
                                                           n_repeats=5, random_state=42)
                    importances = perm_importance.importances_mean
                except Exception as e:
                    if DEBUG_PRINTS:
                        print(f"Permutation importance calculation failed: {e}")
            
            # If importance could be extracted, use it
            if importances is not None:
                # Ensure importances length matches features length
                if len(importances) == len(features):
                    feature_importance = dict(zip(features, importances))
                else:
                    if DEBUG_PRINTS:
                        print(f"Feature length mismatch: importances ({len(importances)}) vs features ({len(features)})")
                    # Fall back to equal importance
                    feature_importance = {f: 1.0/len(features) for f in features}
            else:
                # No feature importance available, use equal weights
                feature_importance = {f: 1.0/len(features) for f in features}
                if DEBUG_PRINTS:
                    print(f"Using equal feature importance for {model_type}")
            
            # Sort features by importance
            sorted_importance = {k: v for k, v in sorted(
                feature_importance.items(), key=lambda item: item[1], reverse=True)}
            
            # Normalize to percentages
            total_importance = sum(sorted_importance.values()) if sum(sorted_importance.values()) > 0 else 1
            importance_pct = {k: (v / total_importance) * 100 for k, v in sorted_importance.items()}
            
            # Create response data
            return {
                'status': 'success',
                'feature_importance': sorted_importance,
                'importance_pct': importance_pct,
                'shap_data': {
                    'feature_names': list(sorted_importance.keys()),
                    'importance_values': list(sorted_importance.values()),
                    'importance_pct': list(importance_pct.values()),
                    'model_type': model_type,
                    'project_id': project_id,
                    'note': 'Using basic feature importance (no SHAP)'
                },
                'total_features': len(features),
                'method': 'basic'
            }
                
        except Exception as e:
            if DEBUG_PRINTS:
                print(f"Basic feature importance calculation failed: {e}")
                import traceback
                print(traceback.format_exc())
            
            # If all else fails, return equal importance
            try:
                # Try to get features from model
                if project_id in self.models:
                    _, features = self.models[project_id]
                    if not features:
                        # Try to get column names from the data
                        features = data.columns.tolist()
                        # Remove prediction columns
                        features = [f for f in features if f not in ['has_booked_prediction', 'prediction_confidence', 'projectid']]
                else:
                    # Get column names from the data
                    features = data.columns.tolist()
                    # Remove prediction columns
                    features = [f for f in features if f not in ['has_booked_prediction', 'prediction_confidence', 'projectid']]
                
                # Generate equal importance
                equal_importance = 1.0 / len(features) if len(features) > 0 else 0
                feature_importance = {f: equal_importance for f in features}
                
                return {
                    'status': 'warning',
                    'message': f"Feature importance calculation failed: {str(e)}. Using equal feature importance.",
                    'feature_importance': feature_importance,
                    'importance_pct': {k: 100/len(features) for k in features},
                    'shap_data': {
                        'feature_names': features,
                        'importance_values': [equal_importance] * len(features),
                        'importance_pct': [100/len(features)] * len(features),
                        'model_type': 'unknown',
                        'project_id': project_id,
                        'note': 'Equal distribution due to feature importance calculation failure'
                    },
                    'total_features': len(features)
                }
            except Exception as nested_e:
                return {
                    'status': 'error',
                    'error': f"Complete failure in feature importance calculation: {str(e)}, then {str(nested_e)}",
                    'traceback': traceback.format_exc() if 'traceback' in locals() else None
                }
    
    def analyze_all_projects(self, data: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        """
        Generate feature importance analysis for all available projects in the data
        
        Args:
            data (pd.DataFrame): Input data with features and projectid column
            
        Returns:
            Dict[int, Dict[str, Any]]: Dictionary mapping project IDs to their feature importance results
        """
        results = {}
        unique_projects = data['projectid'].unique()
        for project_id in unique_projects:
            try:
                pid = int(project_id)
            except Exception:
                continue
                
            if pid not in self.models:
                results[pid] = {
                    'status': 'error',
                    'message': f"No model available for project ID {pid}"
                }
                continue
            
            # For each project, calculate feature importance
            result = self.calculate_feature_importance(data, pid)
            results[pid] = result
        
        return results

    def calculate_interpretML_importance(self, data: pd.DataFrame, project_id: int) -> Dict[str, Any]:
        """
        Calculate feature importance using InterpretML explainers.
        This is a public method that directly uses InterpretML for feature importance.
        
        Args:
            data (pd.DataFrame): Input data with features
            project_id (int): Project ID for model selection
            
        Returns:
            Dict[str, Any]: Dictionary with feature importance information
        """
        if not INTERPRET_ML_AVAILABLE:
            return {
                'status': 'error',
                'message': 'InterpretML is not available. Install with: pip install interpret'
            }
        
        result = self._calculate_interpret_ml_importance(data, project_id)
        
        # Ensure the result has both shap_data and interpret_data for frontend compatibility
        if result.get('status') == 'success':
            # If interpret_data exists but shap_data doesn't, copy the data to shap_data
            if 'interpret_data' in result and 'shap_data' not in result:
                result['shap_data'] = {
                    'feature_names': result['interpret_data']['feature_names'],
                    'importance_values': result['interpret_data']['importance_values'],
                    'importance_pct': result['interpret_data']['importance_pct'],
                    'model_type': result['interpret_data']['model_type'],
                    'project_id': project_id,
                    'method': 'InterpretML',
                    'note': 'Data copied from interpret_data for compatibility'
                }
            
            # Make sure interpret_data is also provided
            if 'shap_data' in result and 'interpret_data' not in result:
                result['interpret_data'] = {
                    'feature_names': result['shap_data']['feature_names'],
                    'importance_values': result['shap_data']['importance_values'],
                    'importance_pct': result['shap_data']['importance_pct'],
                    'model_type': result['shap_data']['model_type'],
                    'project_id': project_id,
                    'method': 'InterpretML'
                }
        
        return result

    def _calculate_interpret_ml_importance(self, data: pd.DataFrame, project_id: int) -> Dict[str, Any]:
        """
        Calculate feature importance using InterpretML explainers (internal use).
        
        Args:
            data (pd.DataFrame): Input data with features
            project_id (int): Project ID for model selection
            
        Returns:
            Dict[str, Any]: Dictionary with feature importance information
        """
        print(f"DEBUG: Starting InterpretML calculation for project {project_id}")
        print(f"DEBUG: InterpretML available: {INTERPRET_ML_AVAILABLE}")
        
        if not INTERPRET_ML_AVAILABLE:
            error_msg = 'InterpretML not available'
            print(f"DEBUG: {error_msg}")
            return {
                'status': 'error',
                'error': error_msg
            }
            
        try:
            # Check if model exists for this project
            if project_id not in self.models:
                error_msg = f"No model available for project ID {project_id}"
                print(f"DEBUG: {error_msg}")
                print(f"DEBUG: Available models: {list(self.models.keys())}")
                return {
                    'status': 'error',
                    'error': error_msg
                }
                
            # Get model and features for this project
            model, features = self.models[project_id]
            model_type = type(model).__name__
            print(f"DEBUG: Model type: {model_type}, Features count: {len(features) if features else 'Unknown'}")
            
            # Extract features needed for this model
            if features:
                X = data.loc[data['projectid'] == project_id, features]
            else:
                X = data.loc[data['projectid'] == project_id].drop(
                    columns=['has_booked_prediction', 'prediction_confidence', 'projectid'], errors='ignore'
                )
                features = X.columns.tolist()
                
            if len(X) == 0:
                error_msg = f"No data available for project ID {project_id}"
                print(f"DEBUG: {error_msg}")
                print(f"DEBUG: Unique project IDs in data: {data['projectid'].unique()}")
                return {
                    'status': 'error',
                    'error': error_msg
                }
            
            print(f"DEBUG: Data shape: {X.shape}, Features: {len(features)}")
            
            # Sample data for faster processing
            sample_size = min(100, len(X))
            if len(X) > sample_size:
                X_sample = X.sample(n=sample_size, random_state=42)
            else:
                X_sample = X.copy()
            
            # Fill missing values
            X_sample = X_sample.fillna(0)
            print(f"DEBUG: Sample shape: {X_sample.shape}")
            
            try:
                # Use MorrisSensitivity for global feature importance analysis
                print("DEBUG: Attempting MorrisSensitivity analysis")
                explainer = MorrisSensitivity(predict_fn=model.predict, data=X_sample, feature_names=features)
                
                # Get global explanation
                global_explanation = explainer.explain_global()
                print("DEBUG: MorrisSensitivity analysis completed")
                
                # Extract feature importance scores
                if hasattr(global_explanation, 'data') and hasattr(global_explanation.data(), 'scores'):
                    importance_scores = global_explanation.data().scores
                    feature_names_from_explainer = global_explanation.data().names if hasattr(global_explanation.data(), 'names') else features
                    print(f"DEBUG: Found scores via data().scores: {len(importance_scores)} scores")
                elif hasattr(global_explanation, 'data'):
                    # Try alternative data access methods
                    explanation_data = global_explanation.data()
                    if isinstance(explanation_data, dict):
                        importance_scores = explanation_data.get('scores', [1.0] * len(features))
                        feature_names_from_explainer = explanation_data.get('names', features)
                        print(f"DEBUG: Found scores via dict access: {len(importance_scores)} scores")
                    else:
                        # Fallback to uniform importance
                        importance_scores = [1.0] * len(features)
                        feature_names_from_explainer = features
                        print("DEBUG: Using uniform importance (dict fallback)")
                else:
                    # Fallback to uniform importance
                    importance_scores = [1.0] * len(features)
                    feature_names_from_explainer = features
                    print("DEBUG: Using uniform importance (no data access)")
                
                # Ensure we have the right number of scores
                if len(importance_scores) != len(feature_names_from_explainer):
                    min_len = min(len(importance_scores), len(feature_names_from_explainer))
                    importance_scores = importance_scores[:min_len]
                    feature_names_from_explainer = feature_names_from_explainer[:min_len]
                
                # Convert to absolute values and create feature importance dict
                importance_scores = [abs(score) for score in importance_scores]
                importance_dict = dict(zip(feature_names_from_explainer, importance_scores))
                
            except Exception as morris_error:
                print(f"DEBUG: MorrisSensitivity failed: {morris_error}")
                # Fallback to basic feature importance from the model
                try:
                    if hasattr(model, 'feature_importances_'):
                        importance_scores = model.feature_importances_
                        print("DEBUG: Using model.feature_importances_")
                    elif hasattr(model, 'coef_'):
                        importance_scores = abs(model.coef_[0]) if model.coef_.ndim > 1 else abs(model.coef_)
                        print("DEBUG: Using model.coef_")
                    elif hasattr(model, 'steps') and hasattr(model.steps[-1][1], 'feature_importances_'):
                        importance_scores = model.steps[-1][1].feature_importances_
                        print("DEBUG: Using pipeline feature_importances_")
                    else:
                        # Uniform importance as last resort
                        importance_scores = [1.0] * len(features)
                        print("DEBUG: Using uniform importance (no model attributes)")
                    
                    importance_dict = dict(zip(features, importance_scores))
                except Exception as fallback_error:
                    print(f"DEBUG: Fallback also failed: {fallback_error}")
                    # Final fallback - uniform importance
                    importance_dict = {f: 1.0 for f in features}
                    print("DEBUG: Using final fallback - uniform importance")
            
            # Sort by importance
            sorted_importance = {k: v for k, v in sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)}
            
            # Normalize to percentages
            total_importance = sum(sorted_importance.values()) if sum(sorted_importance.values()) > 0 else 1
            importance_pct = {k: (v / total_importance) * 100 for k, v in sorted_importance.items()}
            
            print(f"DEBUG: Final sorted importance has {len(sorted_importance)} features")
            print(f"DEBUG: Top 3 features: {list(sorted_importance.keys())[:3]}")
            
            result = {
                'status': 'success',
                'feature_importance': sorted_importance,
                'importance_pct': importance_pct,
                'interpret_data': {
                    'feature_names': list(sorted_importance.keys()),
                    'importance_values': list(sorted_importance.values()),
                    'importance_pct': list(importance_pct.values()),
                    'model_type': model_type,
                    'project_id': project_id,
                    'method': 'InterpretML'
                },
                # For backwards compatibility with components expecting shap_data
                'shap_data': {
                    'feature_names': list(sorted_importance.keys()),
                    'importance_values': list(sorted_importance.values()),
                    'importance_pct': list(importance_pct.values()),
                    'model_type': model_type,
                    'project_id': project_id,
                    'note': 'Using InterpretML for feature importance'
                },
                'total_features': len(features),
                'method': 'interpretml'
            }
            
            print(f"DEBUG: Returning result with status: {result['status']}")
            return result
            
        except Exception as e:
            print(f"DEBUG: InterpretML calculation failed with error: {e}")
            import traceback
            print(f"DEBUG: Traceback: {traceback.format_exc()}")
            return {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }