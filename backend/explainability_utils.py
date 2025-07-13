"""
Explainability utilities for ML model analysis.

This module provides functions to generate SHAP and InterpretML explanations
for trained ML models in the pipeline.
"""

import warnings
from typing import Dict, Any, List, Union
import pandas as pd
import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")

try:
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn")


def _extract_feature_names_from_preprocessor(preprocessor, input_feature_names: List[str]) -> List[str]:
    """
    Extract feature names after preprocessing, including one-hot encoded features.
    
    Args:
        preprocessor: The fitted preprocessor from the pipeline
        input_feature_names: Original feature names before preprocessing
        
    Returns:
        List of feature names after preprocessing transformations
    """
    try:
        # Try to get feature names directly if available (sklearn >= 1.0)
        if hasattr(preprocessor, 'get_feature_names_out'):
            return list(preprocessor.get_feature_names_out(input_feature_names))
    except Exception:
        pass
    
    # Fallback: manually extract feature names from ColumnTransformer
    feature_names = []
    
    if hasattr(preprocessor, 'transformers_'):
        for name, transformer, columns in preprocessor.transformers_:
            if name == 'remainder':
                continue
                
            if isinstance(columns, slice):
                # Convert slice to actual column names
                actual_columns = input_feature_names[columns]
            else:
                # Handle list of column names or indices
                if isinstance(columns[0], str):
                    actual_columns = columns
                else:
                    actual_columns = [input_feature_names[i] for i in columns]
            
            # Handle different transformer types
            if hasattr(transformer, 'get_feature_names_out'):
                try:
                    trans_feature_names = transformer.get_feature_names_out(actual_columns)
                    feature_names.extend(trans_feature_names)
                except Exception:
                    # Fallback for transformers without proper get_feature_names_out
                    if isinstance(transformer, OneHotEncoder):
                        # Handle OneHotEncoder manually
                        for i, col in enumerate(actual_columns):
                            if hasattr(transformer, 'categories_'):
                                categories = transformer.categories_[i] if len(transformer.categories_) > i else []
                                for category in categories:
                                    feature_names.append(f"{col}_{category}")
                            else:
                                feature_names.append(col)
                    else:
                        # For other transformers, just use original column names
                        feature_names.extend(actual_columns)
            else:
                # Transformer doesn't have get_feature_names_out, use original names
                feature_names.extend(actual_columns)
    else:
        # Preprocessor is not a ColumnTransformer, return original names
        feature_names = input_feature_names.copy()
    
    return feature_names


def generate_shap_explanation(
    model: 'Pipeline',
    df: pd.DataFrame,
    projectid: int
) -> Dict[str, Any]:
    """SHAP analysis with enhanced debugging"""
    
    print(f"[SHAP DEBUG] Starting analysis for project {projectid}")
    print(f"[SHAP DEBUG] Input DataFrame shape: {df.shape}")
    print(f"[SHAP DEBUG] Input DataFrame columns: {list(df.columns)}")
    print(f"[SHAP DEBUG] Unique project IDs in data: {df['projectid'].unique()[:10]}")
    
    if not SHAP_AVAILABLE:
        return {
            'status': 'error',
            'message': 'SHAP library not available. Please install with: pip install shap',
            'feature_names': [],
            'shap_values': [],
            'expected_value': [],
            'raw_data': []
        }
    
    try:
        # Step 1: Filter data for the specific project
        print(f"[SHAP DEBUG] Filtering for project {projectid}")
        project_mask = df['projectid'] == projectid
        
        if not project_mask.any():
            print(f"[SHAP DEBUG] No data found for project {projectid}")
            print(f"[SHAP DEBUG] Available projects: {sorted(df['projectid'].unique())}")
            return {
                'status': 'error',
                'message': f'No data found for project ID {projectid}. Available projects: {sorted(df["projectid"].unique())[:10]}',
                'feature_names': [],
                'shap_values': [],
                'expected_value': [],
                'raw_data': []
            }
        
        X_df = df[project_mask].drop(columns=['has_booked'], errors='ignore').copy()
        print(f"[SHAP DEBUG] Filtered data shape: {X_df.shape}")
        
        if len(X_df) == 0:
            return {
                'status': 'error',
                'message': f'No valid samples found for project ID {projectid}',
                'feature_names': [],
                'shap_values': [],
                'expected_value': [],
                'raw_data': []
            }
        
        # Step 2: Transform with the pipeline's preprocessor
        if 'preprocessor' not in model.named_steps:
            return {
                'status': 'error',
                'message': 'Model pipeline does not contain a preprocessor step',
                'feature_names': [],
                'shap_values': [],
                'expected_value': [],
                'raw_data': []
            }
        
        preprocessor = model.named_steps['preprocessor']
        
        # Get original feature names (excluding target and ID columns)
        original_features = [col for col in X_df.columns if col not in ['projectid', 'has_booked']]
        X_features = X_df[original_features]
        
        # Transform the features
        X_transformed = preprocessor.transform(X_features)
        
        # Step 3: Extract feature names post-preprocessing
        feature_names = _extract_feature_names_from_preprocessor(preprocessor, original_features)
        
        # Ensure feature names match transformed data dimensions
        if hasattr(X_transformed, 'shape') and len(feature_names) != X_transformed.shape[1]:
            # Fallback: create generic feature names if mismatch
            feature_names = [f'feature_{i}' for i in range(X_transformed.shape[1])]
        
        # Step 4: Build SHAP explainer on the classifier step
        if 'classifier' not in model.named_steps:
            return {
                'status': 'error',
                'message': 'Model pipeline does not contain a classifier step',
                'feature_names': feature_names,
                'shap_values': [],
                'expected_value': [],
                'raw_data': X_df.to_dict(orient='records')
            }
        
        classifier = model.named_steps['classifier']
        
        # Create SHAP explainer with appropriate method based on classifier type
        try:
            # Try TreeExplainer first (fastest for tree-based models)
            if hasattr(classifier, 'estimators_') or hasattr(classifier, 'tree_'):
                explainer = shap.TreeExplainer(classifier)
            # Try LinearExplainer for linear models
            elif hasattr(classifier, 'coef_'):
                explainer = shap.LinearExplainer(classifier, X_transformed)
            # Fallback to general Explainer
            else:
                explainer = shap.Explainer(classifier)
                
        except Exception as explainer_error:
            # If specific explainers fail, use general Explainer
            try:
                explainer = shap.Explainer(classifier)
            except Exception as general_error:
                return {
                    'status': 'error',
                    'message': f'Failed to create SHAP explainer: {str(general_error)}',
                    'feature_names': feature_names,
                    'shap_values': [],
                    'expected_value': [],
                    'raw_data': X_df.to_dict(orient='records')
                }
        
        # Step 5: Calculate SHAP values
        try:
            shap_result = explainer(X_transformed)
            
            # Attach feature names for frontend compatibility
            if hasattr(shap_result, 'feature_names'):
                shap_result.feature_names = feature_names
            
        except Exception as shap_error:
            return {
                'status': 'error',
                'message': f'Failed to calculate SHAP values: {str(shap_error)}',
                'feature_names': feature_names,
                'shap_values': [],
                'expected_value': [],
                'raw_data': X_df.to_dict(orient='records')
            }
        
        # Step 6: Extract and format results
        try:
            # Handle different SHAP output formats
            if hasattr(shap_result, 'values'):
                shap_values = shap_result.values
            else:
                shap_values = shap_result
                
            if hasattr(shap_result, 'base_values'):
                base_values = shap_result.base_values
            else:
                # Fallback: use zeros if base values not available
                base_values = np.zeros(len(X_transformed))
            
            # Handle multi-class output (take positive class for binary classification)
            if len(shap_values.shape) == 3 and shap_values.shape[2] == 2:
                shap_values = shap_values[:, :, 1]  # Take positive class
                if len(base_values.shape) == 2:
                    base_values = base_values[:, 1]
            elif len(shap_values.shape) == 3:
                # For multi-class, take the last class or reshape appropriately
                shap_values = shap_values[:, :, -1]
                if len(base_values.shape) == 2:
                    base_values = base_values[:, -1]
            
            # Convert to lists for JSON serialization
            shap_values_list = shap_values.tolist() if hasattr(shap_values, 'tolist') else list(shap_values)
            expected_value_list = base_values.tolist() if hasattr(base_values, 'tolist') else list(base_values)
            
        except Exception as format_error:
            return {
                'status': 'error',
                'message': f'Failed to format SHAP results: {str(format_error)}',
                'feature_names': feature_names,
                'shap_values': [],
                'expected_value': [],
                'raw_data': X_df.to_dict(orient='records')
            }
        
        # Prepare raw data for frontend
        raw_data = X_df.to_dict(orient='records')
        
        # Clean raw data for JSON serialization
        cleaned_raw_data = []
        for record in raw_data:
            cleaned_record = {}
            for key, value in record.items():
                if pd.isna(value):
                    cleaned_record[key] = None
                elif isinstance(value, (np.integer, np.int64)):
                    cleaned_record[key] = int(value)
                elif isinstance(value, (np.floating, np.float64)):
                    cleaned_record[key] = float(value)
                else:
                    cleaned_record[key] = str(value)
            cleaned_raw_data.append(cleaned_record)
        
        return {
            'status': 'success',
            'message': f'SHAP analysis completed for project {projectid} with {len(shap_values_list)} samples',
            'feature_names': feature_names,
            'shap_values': shap_values_list,
            'expected_value': expected_value_list,
            'raw_data': cleaned_raw_data,
            'n_samples': len(shap_values_list),
            'n_features': len(feature_names)
        }
        
    except Exception as e:
        print(f"[SHAP DEBUG] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'status': 'error',
            'message': f'Unexpected error in SHAP analysis: {str(e)}',
            'feature_names': [],
            'shap_values': [],
            'expected_value': [],
            'raw_data': []
        }


def generate_interpretml_explanation(
    model: 'Pipeline',
    df: pd.DataFrame,
    projectid: int
) -> Dict[str, Any]:
    """
    Placeholder for InterpretML explanation function.
    
    This function will be implemented to provide InterpretML-based explanations
    as an alternative to SHAP analysis.
    
    Args:
        model: trained Pipeline with steps ['preprocessor', 'classifier']
        df: preprocessed DataFrame
        projectid: project ID for analysis
        
    Returns:
        Dict with explanation results (to be implemented)
    """
    return {
        'status': 'error',
        'message': 'InterpretML explanation not yet implemented',
        'feature_names': [],
        'shap_values': [],
        'expected_value': [],
        'raw_data': []
    }