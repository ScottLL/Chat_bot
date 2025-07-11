"""
Cache manager for storing and retrieving embeddings and other computed data.

This module provides efficient caching mechanisms to avoid recomputing
expensive operations like OpenAI embeddings in the DeFi Q&A chatbot.
"""

import os
import json
import pickle
import hashlib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
from datetime import datetime, timedelta


class CacheManager:
    """Manages caching of embeddings and other computed data."""
    
    def __init__(
        self, 
        cache_dir: str = "./cache",
        max_age_days: int = 30,
        use_compression: bool = True
    ):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Directory to store cache files
            max_age_days: Maximum age of cache files before expiry
            use_compression: Whether to use compression for large data
        """
        self.cache_dir = Path(cache_dir)
        self.max_age_days = max_age_days
        self.use_compression = use_compression
        self._cache_available = False
        
        # Try to create cache directory and test permissions
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Test write permissions by creating a test file
            test_file = self.cache_dir / ".cache_test"
            try:
                test_file.write_text("test")
                test_file.unlink()  # Clean up test file
                self._cache_available = True
                print(f"✅ Cache directory initialized: {self.cache_dir}")
            except PermissionError:
                print(f"⚠️  Cache directory exists but no write permissions: {self.cache_dir}")
                self._cache_available = False
            except Exception as e:
                print(f"⚠️  Cache test failed: {e}")
                self._cache_available = False
                
        except PermissionError as e:
            print(f"⚠️  Cannot create cache directory due to permissions: {self.cache_dir}")
            print(f"    Error: {e}")
            print("    Cache operations will be disabled")
            self._cache_available = False
        except Exception as e:
            print(f"⚠️  Cache initialization failed: {e}")
            print("    Cache operations will be disabled")
            self._cache_available = False
        
    def _get_cache_path(self, key: str, extension: str = ".cache") -> Path:
        """Get the file path for a cache key."""
        # Create a safe filename from the key
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{safe_key}{extension}"
        
    def _is_cache_valid(self, file_path: Path) -> bool:
        """Check if a cache file is still valid based on age."""
        if not file_path.exists():
            return False
            
        # Check file age
        file_age = datetime.now() - datetime.fromtimestamp(file_path.stat().st_mtime)
        return file_age < timedelta(days=self.max_age_days)
        
    def _generate_data_hash(self, data: Any) -> str:
        """Generate a hash for data integrity checking."""
        if isinstance(data, (list, dict)):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        return hashlib.sha256(data_str.encode()).hexdigest()
        
    def save_embeddings(
        self, 
        key: str, 
        embeddings: List[List[float]], 
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Save embeddings to cache with metadata.
        
        Args:
            key: Unique identifier for the embeddings
            embeddings: List of embedding vectors
            metadata: Optional metadata about the embeddings
            
        Returns:
            bool: True if saved successfully
        """
        if not self._cache_available:
            print("⚠️  Cache is not available. Cannot save embeddings.")
            return False
        try:
            cache_data = {
                "embeddings": embeddings,
                "metadata": metadata or {},
                "timestamp": time.time(),
                "data_hash": self._generate_data_hash(embeddings),
                "embedding_count": len(embeddings),
                "embedding_dim": len(embeddings[0]) if embeddings else 0
            }
            
            cache_path = self._get_cache_path(key, ".embeddings")
            
            # Save as pickle for efficiency with numpy arrays
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
                
            return True
            
        except Exception as e:
            print(f"Error saving embeddings to cache: {e}")
            return False
            
    def load_embeddings(self, key: str) -> Optional[Dict]:
        """
        Load embeddings from cache.
        
        Args:
            key: Unique identifier for the embeddings
            
        Returns:
            Dict with embeddings and metadata, or None if not found/invalid
        """
        if not self._cache_available:
            print("⚠️  Cache is not available. Cannot load embeddings.")
            return None
        try:
            cache_path = self._get_cache_path(key, ".embeddings")
            
            if not self._is_cache_valid(cache_path):
                return None
                
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                
            # Verify data integrity
            current_hash = self._generate_data_hash(cache_data["embeddings"])
            if current_hash != cache_data.get("data_hash", ""):
                print("Warning: Cache data integrity check failed")
                return None
                
            return cache_data
            
        except Exception as e:
            print(f"Error loading embeddings from cache: {e}")
            return None
            
    def save_json(self, key: str, data: Any) -> bool:
        """
        Save JSON-serializable data to cache.
        
        Args:
            key: Unique identifier for the data
            data: JSON-serializable data to save
            
        Returns:
            bool: True if saved successfully
        """
        if not self._cache_available:
            print("⚠️  Cache is not available. Cannot save JSON data.")
            return False
        try:
            cache_data = {
                "data": data,
                "timestamp": time.time(),
                "data_hash": self._generate_data_hash(data)
            }
            
            cache_path = self._get_cache_path(key, ".json")
            
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
            return True
            
        except Exception as e:
            print(f"Error saving JSON to cache: {e}")
            return False
            
    def load_json(self, key: str) -> Optional[Any]:
        """
        Load JSON data from cache.
        
        Args:
            key: Unique identifier for the data
            
        Returns:
            The cached data, or None if not found/invalid
        """
        if not self._cache_available:
            print("⚠️  Cache is not available. Cannot load JSON data.")
            return None
        try:
            cache_path = self._get_cache_path(key, ".json")
            
            if not self._is_cache_valid(cache_path):
                return None
                
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
                
            # Verify data integrity
            current_hash = self._generate_data_hash(cache_data["data"])
            if current_hash != cache_data.get("data_hash", ""):
                print("Warning: Cache data integrity check failed")
                return None
                
            return cache_data["data"]
            
        except Exception as e:
            print(f"Error loading JSON from cache: {e}")
            return None
            
    def clear_cache(self, pattern: Optional[str] = None) -> int:
        """
        Clear cache files.
        
        Args:
            pattern: Optional pattern to match filenames (default: all cache files)
            
        Returns:
            int: Number of files deleted
        """
        if not self._cache_available:
            print("⚠️  Cache is not available. Cannot clear cache.")
            return 0
        try:
            files_deleted = 0
            
            for file_path in self.cache_dir.iterdir():
                if file_path.is_file():
                    if pattern is None or pattern in file_path.name:
                        file_path.unlink()
                        files_deleted += 1
                        
            return files_deleted
            
        except Exception as e:
            print(f"Error clearing cache: {e}")
            return 0
            
    def cleanup_expired(self) -> int:
        """
        Remove expired cache files.
        
        Returns:
            int: Number of files deleted
        """
        if not self._cache_available:
            print("⚠️  Cache is not available. Cannot clean up expired cache.")
            return 0
        try:
            files_deleted = 0
            
            for file_path in self.cache_dir.iterdir():
                if file_path.is_file() and not self._is_cache_valid(file_path):
                    file_path.unlink()
                    files_deleted += 1
                    
            return files_deleted
            
        except Exception as e:
            print(f"Error cleaning up expired cache: {e}")
            return 0
            
    def get_cache_stats(self) -> Dict:
        """
        Get statistics about the cache.
        
        Returns:
            Dict with cache statistics
        """
        if not self._cache_available:
            print("⚠️  Cache is not available. Cannot get cache stats.")
            return {}
        try:
            stats = {
                "total_files": 0,
                "total_size_mb": 0,
                "embedding_files": 0,
                "json_files": 0,
                "expired_files": 0,
                "cache_dir": str(self.cache_dir)
            }
            
            total_size = 0
            
            for file_path in self.cache_dir.iterdir():
                if file_path.is_file():
                    stats["total_files"] += 1
                    file_size = file_path.stat().st_size
                    total_size += file_size
                    
                    if file_path.suffix == ".embeddings":
                        stats["embedding_files"] += 1
                    elif file_path.suffix == ".json":
                        stats["json_files"] += 1
                        
                    if not self._is_cache_valid(file_path):
                        stats["expired_files"] += 1
                        
            stats["total_size_mb"] = round(total_size / (1024 * 1024), 2)
            
            return stats
            
        except Exception as e:
            print(f"Error getting cache stats: {e}")
            return {}


if __name__ == "__main__":
    # Test the cache manager
    cache = CacheManager(cache_dir="./test_cache")
    
    # Test embedding cache
    test_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    test_metadata = {"model": "text-embedding-3-small", "dataset": "defi_qa"}
    
    # Save and load test
    if cache.save_embeddings("test_embeddings", test_embeddings, test_metadata):
        print("✅ Embeddings saved successfully")
        
        loaded = cache.load_embeddings("test_embeddings")
        if loaded:
            print("✅ Embeddings loaded successfully")
            print(f"   - Embedding count: {loaded['embedding_count']}")
            print(f"   - Embedding dimension: {loaded['embedding_dim']}")
            print(f"   - Metadata: {loaded['metadata']}")
        else:
            print("❌ Failed to load embeddings")
    else:
        print("❌ Failed to save embeddings")
    
    # Test JSON cache
    test_data = {"questions": ["What is DeFi?"], "processed": True}
    if cache.save_json("test_data", test_data):
        print("✅ JSON data saved successfully")
        
        loaded_data = cache.load_json("test_data")
        if loaded_data:
            print("✅ JSON data loaded successfully")
            print(f"   - Data: {loaded_data}")
        else:
            print("❌ Failed to load JSON data")
    else:
        print("❌ Failed to save JSON data")
    
    # Display cache stats
    stats = cache.get_cache_stats()
    print(f"\n📊 Cache Statistics:")
    for key, value in stats.items():
        print(f"   - {key}: {value}") 