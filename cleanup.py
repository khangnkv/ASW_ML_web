#!/usr/bin/env python3
"""
Cleanup script to remove unnecessary files and directories
for a cleaner ML Prediction System project structure
"""

import os
import shutil
from pathlib import Path

def remove_path(path, description):
    """Remove a file or directory with confirmation"""
    if path.exists():
        try:
            if path.is_file():
                path.unlink()
            else:
                shutil.rmtree(path)
            print(f"✅ Removed: {description}")
            return True
        except Exception as e:
            print(f"❌ Failed to remove {description}: {e}")
            return False
    else:
        print(f"⚠️  Not found: {description}")
        return True

def main():
    """Main cleanup function"""
    print("🧹 ML Prediction System Cleanup")
    print("=" * 50)
    
    # Files and directories to remove
    items_to_remove = [
        # Duplicate model directory (root level)
        (Path("model"), "Duplicate model directory (use backend/model/ instead)"),
        
        # Old batch files
        (Path("setup.bat"), "Old setup.bat (replaced by setup.py)"),
        (Path("start_backend.bat"), "Old start_backend.bat (replaced by start_backend.py)"),
        (Path("start_frontend.bat"), "Old start_frontend.bat (replaced by start_frontend.py)"),
        
        # Test files
        (Path("test_connection.py"), "Test file (not needed for production)"),
        (Path("test_retention.py"), "Test file (not needed for production)"),
        (Path("test_models.py"), "Test file (not needed for production)"),
        
        # Sample data files
        (Path("sample_predict.csv"), "Sample data file (not needed for production)"),
        (Path("sample_predict.xlsx"), "Sample data file (not needed for production)"),
        
        # Duplicate uploads directory
        (Path("uploads"), "Duplicate uploads directory (use backend/uploads/ instead)"),
        
        # Cache directories
        (Path("__pycache__"), "Python cache directory"),
        (Path("backend/__pycache__"), "Backend Python cache directory"),
    ]
    
    # Optional documentation files (uncomment if you want to remove them)
    optional_items = [
        # (Path("PROJECT_FILTERING_README.md"), "Documentation file (optional)"),
        # (Path("DATA_RETENTION_README.md"), "Documentation file (optional)"),
        # (Path("TROUBLESHOOTING.md"), "Documentation file (optional)"),
    ]
    
    print("📋 Items to be removed:")
    for path, description in items_to_remove:
        print(f"   - {description}")
    
    if optional_items:
        print("\n📋 Optional items (uncomment in script to remove):")
        for path, description in optional_items:
            print(f"   - {description}")
    
    # Confirm before proceeding
    response = input("\n❓ Do you want to proceed with cleanup? (y/N): ").strip().lower()
    if response != 'y':
        print("❌ Cleanup cancelled")
        return
    
    print("\n🧹 Starting cleanup...")
    
    # Remove items
    success_count = 0
    total_count = len(items_to_remove)
    
    for path, description in items_to_remove:
        if remove_path(path, description):
            success_count += 1
    
    print(f"\n✅ Cleanup completed: {success_count}/{total_count} items removed")
    
    # Show remaining structure
    print("\n📁 Remaining project structure:")
    print("   backend/")
    print("   ├── app_workflow.py (main Flask app)")
    print("   ├── model/ (ML models)")
    print("   ├── uploads/ (file uploads)")
    print("   └── data_retention.py")
    print("   frontend/ (React app)")
    print("   preprocessing.py")
    print("   requirements.txt")
    print("   run_system.py (main runner)")
    print("   run_backend_only.py (backend-only runner)")
    print("   setup.py (setup script)")
    print("   start_backend.py")
    print("   start_frontend.py")
    
    print("\n🎯 To run the system:")
    print("   python run_system.py")
    print("   or")
    print("   python run_backend_only.py (backend only)")

if __name__ == "__main__":
    main() 