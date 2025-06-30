#!/usr/bin/env python3
"""
System Integrity Test for ML Prediction System (Simplified)
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test all critical imports"""
    print("🔍 Testing imports...")
    
    try:
        # Test backend imports
        sys.path.append(str(Path(__file__).parent))
        
        # Test basic libraries
        import pandas as pd
        import numpy as np
        import flask
        import sklearn
        print("✅ Basic libraries imported successfully")
        
        # Test model predictor
        from model.predictor import MLPredictor
        predictor = MLPredictor()
        print(f"✅ MLPredictor loaded with {len(predictor.models)} models")
        
        # Test preprocessing
        from preprocessing import preprocess_data, get_raw_preview
        print("✅ Preprocessing functions imported successfully")
        
        # Test data retention
        from backend.data_retention import retention_manager
        stats = retention_manager.get_storage_stats()
        print("✅ Data retention manager working")
        
        # Test performance utils
        from performance_utils import PerformanceMonitor
        monitor = PerformanceMonitor()
        print("✅ Performance utilities imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic file operations"""
    print("\n🔍 Testing basic functionality...")
    
    try:
        import pandas as pd
        
        # Create a test CSV file
        test_data = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        test_file = Path('test_temp.csv')
        
        # Test basic file operations
        test_data.to_csv(test_file, index=False)
        df = pd.read_csv(test_file)
        print(f"✅ Basic file reading: {len(df)} rows")
        
        # Cleanup
        test_file.unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test error: {e}")
        return False

def test_directories():
    """Test required directories exist or can be created"""
    print("\n🔍 Testing directory structure...")
    
    required_dirs = [
        'uploads',
        'model',
        'backend',
        'frontend/src',
        'backend/preprocessed_unencoded'
    ]
    
    try:
        for dir_path in required_dirs:
            path = Path(dir_path)
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                print(f"✅ Created directory: {dir_path}")
            else:
                print(f"✅ Directory exists: {dir_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Directory test error: {e}")
        return False

def test_models():
    """Test model files exist and can be loaded"""
    print("\n🔍 Testing model files...")
    
    try:
        model_dir = Path('model')
        model_files = list(model_dir.glob('*.pkl'))
        
        if not model_files:
            print("⚠️  No model files found - this is expected for new installations")
            return True
        
        print(f"✅ Found {len(model_files)} model files")
        
        # Test loading one model
        import joblib
        test_model = model_files[0]
        model_data = joblib.load(test_model)
        print(f"✅ Successfully loaded test model: {test_model.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 ML Prediction System Integrity Test (Simplified)")
    print("=" * 60)
    
    tests = [
        test_directories,
        test_imports,
        test_basic_functionality,
        test_models
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ All {total} tests passed! System is ready.")
        return 0
    else:
        print(f"❌ {total - passed} out of {total} tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
