# Debug Control Instructions

## How to Turn Off Debug Prints

To disable all debug printing output from the ML backend, you need to set `DEBUG_PRINTS = False` in the following files:

### Files to Modify:

1. **`backend/app_workflow.py`**
   - Line ~15: Change `DEBUG_PRINTS = False`

2. **`backend/model/predictor.py`**
   - Line ~9: Change `DEBUG_PRINTS = False`

3. **`preprocessing.py`**
   - Line ~11: Change `DEBUG_PRINTS = False`

### To Enable Debug Prints:

Set `DEBUG_PRINTS = True` in all the above files.

### What Debug Output is Controlled:

- `[PREDICT]` messages from the prediction workflow
- `[Predict] Processing project_id=X` messages 
- `Predicting for project_id=X, features=[...]` messages
- `Loaded project info from:` messages
- Model loading and error messages
- Preprocessing status messages

### After Making Changes:

1. Save all the modified files
2. Restart your Docker containers or Python backend service
3. The debug output should now be disabled/enabled based on your setting

### Recent Updates:

**Prediction Display Improvements:**
- Added prediction confidence scores alongside predictions
- Updated Final Prediction Preview to show both "Prediction" and "Confidence" columns
- Improved prediction results counting (Likely to book vs Unlikely to book)
- Enhanced table formatting with color coding for predictions

**New Features:**
- Predictions are now displayed as "Likely to book (75.3%)" or "Unlikely to book (24.7%)"
- Confidence scores show the model's certainty in each prediction
- Green highlighting for "likely to book" predictions, red for "unlikely to book"
- Proper counting in Prediction Results statistics

### Note:

- Some warning messages and error messages may still appear as they are important for debugging actual issues
- This only affects the verbose debugging output, not critical error logging
- Prediction columns are automatically moved to the end of the table for better visibility
