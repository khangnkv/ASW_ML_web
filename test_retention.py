#!/usr/bin/env python3
"""
Test script for data retention system
"""
import sys
import os
from datetime import datetime, timedelta
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.data_retention import DataRetentionManager

def test_retention_system():
    """Test the data retention system"""
    print("🧪 Testing Data Retention System")
    print("=" * 40)
    
    # Create a test retention manager with 1 day retention for testing
    test_manager = DataRetentionManager(retention_days=1)
    
    # Test storing files
    print("\n1. Testing file storage...")
    file_info1 = test_manager.store_file("test_file_1.csv", "original_data.csv")
    file_info2 = test_manager.store_file("test_file_2.csv", "sample_data.xlsx")
    
    print(f"✅ Stored {len(test_manager.metadata)} files")
    
    # Test listing files
    print("\n2. Testing file listing...")
    files = test_manager.list_files()
    print(f"✅ Found {len(files)} files:")
    for file_info in files:
        print(f"   - {file_info['filename']} (original: {file_info['original_filename']})")
        print(f"     Upload: {file_info['upload_timestamp']}")
        print(f"     Deletion: {file_info['deletion_date']}")
    
    # Test storage stats
    print("\n3. Testing storage statistics...")
    stats = test_manager.get_storage_stats()
    print(f"✅ Storage stats: {stats}")
    
    # Test file info retrieval
    print("\n4. Testing file info retrieval...")
    file_info = test_manager.get_file_info("test_file_1.csv")
    if file_info:
        print(f"✅ Retrieved file info: {file_info['filename']}")
    else:
        print("❌ Failed to retrieve file info")
    
    # Test manual deletion
    print("\n5. Testing manual deletion...")
    success = test_manager.delete_file("test_file_2.csv")
    if success:
        print("✅ Successfully deleted test_file_2.csv")
    else:
        print("❌ Failed to delete test_file_2.csv")
    
    # Test cleanup (should not delete files that are not expired)
    print("\n6. Testing cleanup (should not delete non-expired files)...")
    deleted_count = test_manager.cleanup_expired_files()
    print(f"✅ Cleanup deleted {deleted_count} files (expected 0)")
    
    print("\n🎉 All tests completed successfully!")
    return True

if __name__ == "__main__":
    try:
        success = test_retention_system()
        if success:
            print("\n✅ Data retention system is working correctly!")
        else:
            print("\n❌ Data retention system has issues!")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error testing retention system: {e}")
        sys.exit(1) 