import os
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import threading
import time

class DataRetentionManager:
    def __init__(self, uploads_dir: str = "uploads", retention_days: int = 90):
        """
        Initialize the data retention manager.
        
        Args:
            uploads_dir: Directory to store uploaded files
            retention_days: Number of days to keep files before deletion
        """
        self.uploads_dir = Path(uploads_dir)
        self.retention_days = retention_days
        self.metadata_file = self.uploads_dir / "file_metadata.json"
        self.metadata: Dict[str, Dict] = {}
        
        # Ensure uploads directory exists
        self.uploads_dir.mkdir(exist_ok=True)
        
        # Load existing metadata
        self._load_metadata()
        
        # Start cleanup thread
        self._start_cleanup_thread()
    
    def _load_metadata(self):
        """Load file metadata from JSON file"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load metadata file: {e}")
                self.metadata = {}
    
    def _save_metadata(self):
        """Save file metadata to JSON file"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except IOError as e:
            print(f"Error saving metadata: {e}")
    
    def store_file(self, filename: str, original_filename: Optional[str] = None) -> Dict:
        """
        Store file metadata with timestamps.
        
        Args:
            filename: The stored filename (UUID)
            original_filename: Original filename from user
            
        Returns:
            Dict with file metadata
        """
        now = datetime.now()
        deletion_date = now + timedelta(days=self.retention_days)
        
        file_info = {
            "filename": filename,
            "original_filename": original_filename or filename,
            "upload_timestamp": now.isoformat(),
            "deletion_date": deletion_date.isoformat(),
            "file_path": str(self.uploads_dir / filename),
            "status": "active"
        }
        
        self.metadata[filename] = file_info
        self._save_metadata()
        
        print(f"📁 File stored: {filename}")
        print(f"   📅 Upload time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   🗑️  Deletion date: {deletion_date.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return file_info
    
    def get_file_info(self, filename: str) -> Optional[Dict]:
        """Get file metadata"""
        return self.metadata.get(filename)
    
    def list_files(self) -> List[Dict]:
        """List all files with their metadata"""
        return list(self.metadata.values())
    
    def delete_file(self, filename: str) -> bool:
        """
        Manually delete a file and its metadata.
        
        Args:
            filename: The filename to delete
            
        Returns:
            True if successful, False otherwise
        """
        if filename not in self.metadata:
            return False
        
        file_info = self.metadata[filename]
        file_path = Path(file_info["file_path"])
        
        try:
            # Delete the file
            if file_path.exists():
                file_path.unlink()
                print(f"🗑️  Deleted file: {filename}")
            
            # Remove metadata
            del self.metadata[filename]
            self._save_metadata()
            
            return True
        except Exception as e:
            print(f"Error deleting file {filename}: {e}")
            return False
    
    def cleanup_expired_files(self) -> int:
        """
        Remove files that have exceeded the retention period.
        
        Returns:
            Number of files deleted
        """
        now = datetime.now()
        expired_files = []
        
        for filename, file_info in self.metadata.items():
            deletion_date = datetime.fromisoformat(file_info["deletion_date"])
            if now >= deletion_date:
                expired_files.append(filename)
        
        deleted_count = 0
        for filename in expired_files:
            if self.delete_file(filename):
                deleted_count += 1
        
        if deleted_count > 0:
            print(f"🧹 Cleaned up {deleted_count} expired files")
        
        return deleted_count
    
    def get_storage_stats(self) -> Dict:
        """Get storage statistics"""
        total_files = len(self.metadata)
        total_size = 0
        
        for file_info in self.metadata.values():
            file_path = Path(file_info["file_path"])
            if file_path.exists():
                total_size += file_path.stat().st_size
        
        return {
            "total_files": total_files,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "retention_days": self.retention_days
        }
    
    def _start_cleanup_thread(self):
        """Start background thread for periodic cleanup"""
        def cleanup_worker():
            while True:
                try:
                    # Run cleanup every hour
                    time.sleep(3600)
                    self.cleanup_expired_files()
                except Exception as e:
                    print(f"Error in cleanup thread: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        print(f"🔄 Started cleanup thread (runs every hour, retention: {self.retention_days} days)")

# Global instance
retention_manager = DataRetentionManager() 