# Data Retention System

## Overview

The ML Prediction System now includes a comprehensive data retention system that automatically manages uploaded files with a 90-day retention period. This ensures data privacy and efficient storage management.

## Features

### 🔄 **Automatic File Management**
- **90-day retention period**: Files are automatically deleted after 90 days
- **Background cleanup**: Runs every hour to remove expired files
- **Timestamp tracking**: Each file is tracked with upload and deletion dates

### 📊 **File Metadata Tracking**
- **Upload timestamp**: Exact time when file was uploaded
- **Deletion date**: When the file will be automatically deleted
- **Original filename**: Preserves the user's original filename
- **File status**: Tracks if file is active or deleted

### 🛠️ **Management Features**
- **Storage statistics**: View total files and storage usage
- **Manual deletion**: Delete files before expiration if needed
- **File listing**: View all stored files with metadata
- **Manual cleanup**: Trigger immediate cleanup of expired files

## Implementation Details

### Backend Components

#### 1. **DataRetentionManager** (`backend/data_retention.py`)
```python
# Core class that handles all retention logic
retention_manager = DataRetentionManager(retention_days=90)
```

**Key Methods:**
- `store_file(filename, original_filename)` - Store file with metadata
- `cleanup_expired_files()` - Remove expired files
- `get_storage_stats()` - Get storage statistics
- `list_files()` - List all files with metadata

#### 2. **API Endpoints** (`backend/app_workflow.py`)
```python
# New endpoints for retention management
GET /api/storage/stats          # Get storage statistics
GET /api/storage/files          # List all stored files
DELETE /api/storage/files/<id>  # Delete specific file
POST /api/storage/cleanup       # Manual cleanup trigger
```

#### 3. **Integration with Upload Process**
```python
# When file is uploaded, it's automatically tracked
file_info = retention_manager.store_file(uuid_name, file.filename)
```

### Frontend Components

#### 1. **File Retention Information Display**
- Shows upload time and deletion date
- Displays days remaining with color-coded badges
- Green: >30 days, Yellow: 7-30 days, Red: <7 days

#### 2. **Storage Statistics Dashboard**
- Total files stored
- Total storage size in MB
- Retention period (90 days)
- Refresh button to update stats

## File Storage Structure

```
uploads/
├── file_metadata.json          # Metadata for all files
├── {uuid}.csv                  # Actual uploaded files (CSV format)
└── results_{uuid}.csv          # Prediction results
```

### Metadata Format
```json
{
  "filename": "uuid.csv",
  "original_filename": "user_file.xlsx",
  "upload_timestamp": "2025-06-26T08:55:52.809218",
  "deletion_date": "2025-09-24T08:55:52.809218",
  "file_path": "uploads/uuid.csv",
  "status": "active"
}
```

## Usage Examples

### 1. **Upload a File**
When a user uploads a file, the system automatically:
- Converts it to CSV format for faster processing
- Generates a UUID filename
- Stores metadata with timestamps
- Sets deletion date to 90 days from upload

### 2. **View File Information**
The frontend displays:
- Upload time and deletion date
- Days remaining until deletion
- File status (active/deleted)

### 3. **Storage Management**
- View total files and storage usage
- Manually delete files if needed
- Trigger immediate cleanup of expired files

## Configuration

### Retention Period
The retention period can be configured in `backend/data_retention.py`:
```python
retention_manager = DataRetentionManager(retention_days=90)  # Change this value
```

### Cleanup Frequency
The background cleanup runs every hour by default. To change this:
```python
# In DataRetentionManager._start_cleanup_thread()
time.sleep(3600)  # Change from 3600 seconds (1 hour) to desired interval
```

## Security & Privacy

### Data Protection
- Files are stored with UUID names (not original filenames)
- Original filenames are preserved in metadata only
- Automatic deletion ensures no long-term data retention
- Background cleanup prevents manual intervention

### Access Control
- File metadata is stored locally in JSON format
- No external dependencies for data storage
- Cleanup process runs as a daemon thread

## Monitoring & Maintenance

### Logs
The system provides detailed logging:
```
📁 File stored: uuid.csv
   📅 Upload time: 2025-06-26 08:55:52
   🗑️  Deletion date: 2025-09-24 08:55:52
🧹 Cleaned up 3 expired files
```

### Health Checks
- Background cleanup thread runs continuously
- Storage statistics are updated in real-time
- File metadata is automatically saved/loaded

## Testing

Run the test script to verify the system:
```bash
python test_retention.py
```

This will test:
- File storage with metadata
- File listing and retrieval
- Storage statistics
- Manual deletion
- Cleanup functionality

## Benefits

1. **Automatic Management**: No manual intervention required
2. **Data Privacy**: Files are automatically deleted after 90 days
3. **Storage Efficiency**: Prevents unlimited file accumulation
4. **User Transparency**: Users can see when their files will be deleted
5. **Compliance**: Helps meet data retention requirements
6. **Performance**: CSV-only storage improves processing speed

## Future Enhancements

Potential improvements:
- Configurable retention periods per user
- Email notifications before file deletion
- File compression for storage optimization
- Backup and restore functionality
- Advanced analytics on storage usage 