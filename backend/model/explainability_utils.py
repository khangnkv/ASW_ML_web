import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import warnings

def extract_feature_importance(model, project_id: int) -> Dict[str, Any]:
    """
    Extract feature importance from a trained model for a specific project.
    
    Args:
        model: Trained model (from predictor.models[project_id])
        project_id: Project ID for identification
    
    Returns:
        Dictionary with feature importance results
    """
    try:
        # Get the actual estimator from pipeline or model tuple
        if isinstance(model, tuple):
            estimator, feature_names = model
        else:
            estimator = model
            feature_names = []
        
        # Handle ensemble models (VotingClassifier, etc.)
        if hasattr(estimator, 'estimators_'):
            # For VotingClassifier, use first estimator
            if hasattr(estimator, 'estimators') and len(estimator.estimators) > 0:
                estimator_name, estimator = estimator.estimators[0]
                model_type = f"VotingClassifier-{type(estimator).__name__}"
            else:
                estimator = estimator.estimators_[0]
                model_type = f"Ensemble-{type(estimator).__name__}"
        else:
            model_type = type(estimator).__name__
        
        importance_values = None
        importance_type = "unknown"
        
        # Tree-based models (RandomForest, XGBoost, LightGBM, etc.)
        if hasattr(estimator, 'feature_importances_'):
            importance_values = estimator.feature_importances_
            importance_type = "tree_importance"
        
        # Linear models (LogisticRegression, SVM, etc.)
        elif hasattr(estimator, 'coef_'):
            coef = estimator.coef_
            if len(coef.shape) > 1:
                # For multi-class, use absolute values of first class or mean
                importance_values = np.abs(coef[0] if coef.shape[0] == 1 else np.mean(np.abs(coef), axis=0))
            else:
                importance_values = np.abs(coef)
            importance_type = "linear_coef"
        
        else:
            return {
                'success': False,
                'error': f'Model type {model_type} does not support feature importance extraction',
                'model_type': model_type,
                'project_id': project_id
            }
        
        # Ensure we have feature names
        if not feature_names or len(feature_names) == 0:
            feature_names = [f"feature_{i}" for i in range(len(importance_values))]
        
        # Ensure dimensions match
        if len(importance_values) != len(feature_names):
            min_len = min(len(importance_values), len(feature_names))
            importance_values = importance_values[:min_len]
            feature_names = feature_names[:min_len]
        
        # Create feature importance data
        importance_data = []
        max_importance = np.max(importance_values) if len(importance_values) > 0 else 1.0
        
        for i, (feature, importance) in enumerate(zip(feature_names, importance_values)):
            importance_data.append({
                'rank': i + 1,
                'feature': str(feature),
                'importance': float(importance),
                'importance_normalized': float(importance / max_importance) if max_importance > 0 else 0.0
            })
        
        # Sort by importance (descending)
        importance_data.sort(key=lambda x: x['importance'], reverse=True)
        
        # Update ranks after sorting
        for i, item in enumerate(importance_data):
            item['rank'] = i + 1
        
        return {
            'success': True,
            'project_id': project_id,
            'model_type': model_type,
            'importance_type': importance_type,
            'total_features': len(importance_data),
            'feature_importance': importance_data,
            'top_features': importance_data[:10]  # Top 10 features
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'project_id': project_id,
            'model_type': model_type if 'model_type' in locals() else 'unknown'
        }

def calculate_conditional_analysis(data: pd.DataFrame, project_id: int) -> Dict[str, Any]:
    """
    Perform conditional dataset-level analysis for a specific project.
    
    Args:
        data: DataFrame with predictions for the specific project
        project_id: Project ID for identification
    
    Returns:
        Dictionary with conditional analysis results
    """
    try:
        # Check for multiple possible prediction column names
        prediction_col = None
        possible_pred_cols = ['has_booked_prediction', 'has_booked', 'prediction', 'has_booked_pred']
        
        for col in possible_pred_cols:
            if col in data.columns:
                prediction_col = col
                break
        
        if prediction_col is None:
            available_cols = list(data.columns)
            return {
                'success': False,
                'error': f'No prediction column found. Available columns: {available_cols}. Expected one of: {possible_pred_cols}',
                'project_id': project_id
            }
        
        # Convert probabilities to binary predictions (>=0.5 means likely to book)
        binary_predictions = (data[prediction_col] >= 0.5).astype(int)
        data_analysis = data.copy()
        data_analysis['prediction_binary'] = binary_predictions
        
        # Identify categorical columns (exclude prediction and ID columns)
        exclude_cols = ['prediction_binary', prediction_col, 'prediction_confidence', 'projectid', 'has_booked']
        categorical_cols = []
        
        for col in data_analysis.columns:
            if col not in exclude_cols:
                # Consider object, categorical, or low-cardinality numeric columns as categorical
                if (data_analysis[col].dtype == 'object' or 
                    data_analysis[col].dtype.name == 'category' or
                    (data_analysis[col].dtype in ['int64', 'float64'] and data_analysis[col].nunique() <= 20)):
                    categorical_cols.append(col)
        
        # Limit to reasonable number of columns for analysis
        categorical_cols = categorical_cols[:15]  # Analyze top 15 categorical features
        
        conditional_results = {}
        
        for col in categorical_cols:
            try:
                # Skip columns with too many unique values
                unique_values = data_analysis[col].dropna().nunique()
                if unique_values > 50 or unique_values < 2:  # Skip high-cardinality or constant features
                    continue
                
                # Calculate conditional probabilities for this feature
                feature_analysis = calculate_feature_conditional_probs(
                    data_analysis, col, 'prediction_binary', project_id
                )
                
                if feature_analysis['success']:
                    conditional_results[col] = feature_analysis
                    
            except Exception as e:
                print(f"Error analyzing column {col} for project {project_id}: {e}")
                continue
        
        # Calculate overall statistics
        total_samples = len(data_analysis)
        positive_predictions = binary_predictions.sum()
        negative_predictions = total_samples - positive_predictions
        
        overall_stats = {
            'project_id': project_id,
            'total_samples': total_samples,
            'positive_predictions': int(positive_predictions),
            'negative_predictions': int(negative_predictions),
            'positive_rate': float(positive_predictions / total_samples) if total_samples > 0 else 0.0,
            'features_analyzed': len(conditional_results),
            'prediction_column_used': prediction_col  # Add this for debugging
        }
        
        # Create analysis summary
        analysis_summary = create_analysis_summary(conditional_results, overall_stats)
        
        return {
            'success': True,
            'project_id': project_id,
            'overall_stats': overall_stats,
            'feature_analysis': conditional_results,
            'analysis_summary': analysis_summary
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'project_id': project_id
        }

def calculate_feature_conditional_probs(data: pd.DataFrame, feature_col: str, 
                                      prediction_col: str, project_id: int) -> Dict[str, Any]:
    """
    Calculate conditional probabilities and class proportion shifts for a single feature.
    
    Args:
        data: DataFrame with the data
        feature_col: Column name of the feature to analyze
        prediction_col: Column name with binary predictions
        project_id: Project ID for identification
    
    Returns:
        Dictionary with feature analysis results
    """
    try:
        # Handle missing values
        feature_data = data[[feature_col, prediction_col]].dropna()
        
        if len(feature_data) == 0:
            return {
                'success': False, 
                'error': 'No valid data after removing missing values',
                'feature': feature_col,
                'project_id': project_id
            }
        
        # Get value counts for each category
        feature_counts = feature_data[feature_col].value_counts()
        
        # Limit to top categories if too many
        if len(feature_counts) > 20:
            top_categories = feature_counts.head(20).index
            feature_data = feature_data[feature_data[feature_col].isin(top_categories)]
        
        # Calculate conditional probabilities P(prediction=1 | feature=category)
        conditional_probs = []
        overall_positive_rate = feature_data[prediction_col].mean()
        
        for category in feature_data[feature_col].unique():
            category_data = feature_data[feature_data[feature_col] == category]
            
            if len(category_data) == 0:
                continue
            
            positive_count = category_data[prediction_col].sum()
            total_count = len(category_data)
            conditional_prob = positive_count / total_count if total_count > 0 else 0.0
            
            # Calculate proportion of this category in positive/negative predictions
            pos_data = feature_data[feature_data[prediction_col] == 1]
            neg_data = feature_data[feature_data[prediction_col] == 0]
            
            prop_in_positive = len(pos_data[pos_data[feature_col] == category]) / len(pos_data) if len(pos_data) > 0 else 0.0
            prop_in_negative = len(neg_data[neg_data[feature_col] == category]) / len(neg_data) if len(neg_data) > 0 else 0.0
            
            # Calculate lift (how much this category increases/decreases prediction probability)
            lift = conditional_prob / overall_positive_rate if overall_positive_rate > 0 else 0.0
            
            conditional_probs.append({
                'category': str(category),
                'total_count': int(total_count),
                'positive_count': int(positive_count),
                'conditional_prob': float(conditional_prob),
                'prop_in_positive': float(prop_in_positive),
                'prop_in_negative': float(prop_in_negative),
                'lift': float(lift),
                'proportion_shift': float(prop_in_positive - prop_in_negative)
            })
        
        # Sort by conditional probability (descending)
        conditional_probs.sort(key=lambda x: x['conditional_prob'], reverse=True)
        
        return {
            'success': True,
            'feature': feature_col,
            'project_id': project_id,
            'unique_categories': len(conditional_probs),
            'overall_positive_rate': float(overall_positive_rate),
            'conditional_probabilities': conditional_probs,
            'highest_prob_category': conditional_probs[0] if conditional_probs else None,
            'lowest_prob_category': conditional_probs[-1] if conditional_probs else None
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'feature': feature_col,
            'project_id': project_id
        }

def create_analysis_summary(conditional_results: Dict, overall_stats: Dict) -> Dict[str, Any]:
    """
    Create a summary of the most important insights from conditional analysis.
    
    Args:
        conditional_results: Results from conditional analysis
        overall_stats: Overall statistics
    
    Returns:
        Dictionary with analysis summary
    """
    try:
        # Find features with highest predictive power
        high_impact_features = []
        low_impact_features = []
        
        for feature, analysis in conditional_results.items():
            if not analysis['success']:
                continue
            
            conditional_probs = analysis['conditional_probabilities']
            if not conditional_probs:
                continue
            
            # Calculate the range of conditional probabilities for this feature
            prob_values = [item['conditional_prob'] for item in conditional_probs]
            prob_range = max(prob_values) - min(prob_values)
            
            feature_summary = {
                'feature': feature,
                'prob_range': float(prob_range),
                'max_prob': float(max(prob_values)),
                'min_prob': float(min(prob_values)),
                'categories_count': len(conditional_probs),
                'highest_category': conditional_probs[0]['category'],
                'highest_prob': float(conditional_probs[0]['conditional_prob'])
            }
            
            if prob_range > 0.3:  # High impact if range > 30%
                high_impact_features.append(feature_summary)
            elif prob_range < 0.1:  # Low impact if range < 10%
                low_impact_features.append(feature_summary)
        
        # Sort by impact
        high_impact_features.sort(key=lambda x: x['prob_range'], reverse=True)
        low_impact_features.sort(key=lambda x: x['prob_range'])
        
        return {
            'high_impact_features': high_impact_features[:5],  # Top 5
            'low_impact_features': low_impact_features[:5],   # Bottom 5
            'total_features_analyzed': overall_stats['features_analyzed'],
            'baseline_positive_rate': overall_stats['positive_rate'],
            'project_id': overall_stats['project_id']
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'high_impact_features': [],
            'low_impact_features': [],
            'project_id': overall_stats.get('project_id', 'unknown')
        }