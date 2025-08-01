# Debug Control Instructions

## How to Turn Off Debug Prints

To disable all debug printing output from the ML backend, set `DEBUG_PRINTS = False` in the following files:

### Files to Modify:

1. **`backend/app_workflow.py`**
2. **`backend/model/predictor.py`**
3. **`preprocessing.py`**

### To Enable Debug Prints:

Set `DEBUG_PRINTS = True` in all the above files.

### What Debug Output is Controlled:

- Prediction workflow messages
- Model loading and error messages
- Preprocessing status messages

### After Making Changes:

1. Save all modified files
2. Restart your backend service or Docker containers

### Note:

- Some warning/error messages may still appear for critical issues.
- This only affects verbose debugging output, not critical error logging.
