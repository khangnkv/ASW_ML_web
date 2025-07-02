# Feature Importance Analysis

This document explains the feature importance analysis capabilities in the ML Prediction System.

## Overview

The feature importance analysis feature uses SHAP (SHapley Additive exPlanations) to interpret machine learning models by explaining the output of any machine learning model by calculating the contribution of each feature to the prediction. This helps in understanding which features are most influential in the model's decision-making process.

## How It Works

1. **SHAP Values**: The system calculates SHAP values for each feature in the model. SHAP values represent the impact that each feature has on the model prediction.

2. **Model-specific Explainers**: Different types of explainers are used depending on the model type:
   - TreeExplainer for tree-based models like Random Forest and XGBoost
   - LinearExplainer for linear models like Logistic Regression
   - KernelExplainer as a fallback for other model types

3. **Visualization**: The results are visualized using bar charts and pie charts, showing the relative importance of each feature in percentage terms.

## Accessing the Feature

The feature importance analysis can be accessed after generating predictions:

1. Upload your data file
2. Generate predictions
3. Click on "View Feature Importance Analysis" button
4. Select a project ID to analyze

## Understanding the Results

### Feature Importance Chart

The chart shows the top features that influence the model's predictions, with the height of each bar (or size of each pie slice) representing the relative importance of each feature.

### Feature Importance Details Table

The table provides more detailed information about each feature's importance:
- **Feature Name**: The name of the feature
- **Importance (%)**: The relative importance as a percentage
- **Normalized Impact**: A visual representation of the impact

## Technical Implementation

### Backend

The backend implements:
- Feature importance calculation using SHAP
- Support for different model types
- API endpoints to retrieve feature importance data

### Frontend

The frontend provides:
- Interactive visualizations (bar and pie charts)
- Project selection
- Detailed data tables
- Export options for charts

## Requirements

All required dependencies are already included in the main `requirements.txt` file:

```
shap>=0.41.0
interpret>=0.4.4
plotly>=5.3.0
matplotlib>=3.7.0
kaleido>=0.2.1  # For static image export with Plotly
```

Frontend dependencies (included in package.json):
```
chart.js@^4.3.0
react-chartjs-2@^5.2.0
```

## Limitations

1. **Computation Time**: For large datasets or complex models, calculating SHAP values can be time-consuming. The system samples data if there are too many rows.

2. **Model Support**: While SHAP can work with any model, the quality and interpretability of explanations may vary depending on the model type.

3. **Memory Usage**: For very large models or datasets, memory constraints may be an issue.

## Future Enhancements

Planned enhancements include:
- Support for more detailed SHAP visualizations (e.g., SHAP dependency plots)
- Feature correlation analysis
- Model comparison tools
- Downloadable PDF reports
