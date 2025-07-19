"""
Advanced Caching System for Ultra-Cost-Optimized Pipeline
Implements multiple caching strategies to minimize compute costs
"""

import hashlib
import json
import pickle
import time
import logging
from typing import Any, Dict, Optional, List, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import redis
import numpy as np
from pathlib import Path
import threading
from functools import wraps


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    value: Any
    timestamp: datetime
    access_count: int = 0
    last_accessed: datetime = None
    ttl: Optional[int] = None
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        if self.ttl is None:
            return False
        return datetime.now() > (self.timestamp + timedelta(seconds=self.ttl))
    
    def touch(self):
        """Update access metadata"""
        self.access_count += 1
        self.last_accessed = datetime.now()


class RedisCache:
    """Redis-based caching for distributed systems"""
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, password: str = None):
        try:
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=False,  # Keep binary for pickle
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            self.redis_client.ping()
            self.available = True
            logging.info(f"Redis cache connected: {host}:{port}")
        except Exception as e:
            logging.warning(f"Redis cache unavailable: {e}")
            self.available = False
            self.redis_client = None
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        if not self.available:
            return None
        
        try:
            data = self.redis_client.get(key)
            if data is not None:
                return pickle.loads(data)
            return None
        except Exception as e:
            logging.error(f"Redis get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in Redis cache"""
        if not self.available:
            return
        
        try:
            data = pickle.dumps(value)
            if ttl:
                self.redis_client.setex(key, ttl, data)
            else:
                self.redis_client.set(key, data)
        except Exception as e:
            logging.error(f"Redis set error: {e}")
    
    def delete(self, key: str):
        """Delete value from Redis cache"""
        if not self.available:
            return
        
        try:
            self.redis_client.delete(key)
        except Exception as e:
            logging.error(f"Redis delete error: {e}")
    
    def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache"""
        if not self.available:
            return False
        
        try:
            return bool(self.redis_client.exists(key))
        except Exception as e:
            logging.error(f"Redis exists error: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics"""
        if not self.available:
            return {}
        
        try:
            info = self.redis_client.info()
            return {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "total_commands_processed": info.get("total_commands_processed", 0)
            }
        except Exception as e:
            logging.error(f"Redis stats error: {e}")
            return {}


class MemoryCache:
    """In-memory LRU cache with size limits"""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 1024):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: Dict[str, CacheEntry] = {}
        self.lock = threading.Lock()
        self.current_memory = 0
        
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes"""
        try:
            return len(pickle.dumps(obj))
        except Exception:
            # Fallback estimation
            if isinstance(obj, str):
                return len(obj.encode('utf-8'))
            elif isinstance(obj, (list, tuple)):
                return sum(self._estimate_size(item) for item in obj)
            elif isinstance(obj, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in obj.items())
            elif isinstance(obj, np.ndarray):
                return obj.nbytes
            else:
                return 1024  # Default estimate
    
    def _evict_lru(self):
        """Evict least recently used items"""
        while (len(self.cache) >= self.max_size or 
               self.current_memory >= self.max_memory_bytes) and self.cache:
            
            # Find LRU entry
            lru_key = min(
                self.cache.keys(),
                key=lambda k: self.cache[k].last_accessed or self.cache[k].timestamp
            )
            
            entry = self.cache.pop(lru_key)
            self.current_memory -= entry.size_bytes
            
            logging.debug(f"Evicted cache entry: {lru_key}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                if entry.is_expired():
                    del self.cache[key]
                    self.current_memory -= entry.size_bytes
                    return None
                
                entry.touch()
                return entry.value
            
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in memory cache"""
        with self.lock:
            size_bytes = self._estimate_size(value)
            
            # Remove existing entry if present
            if key in self.cache:
                old_entry = self.cache[key]
                self.current_memory -= old_entry.size_bytes
            
            # Create new entry
            entry = CacheEntry(
                value=value,
                timestamp=datetime.now(),
                ttl=ttl,
                size_bytes=size_bytes,
                last_accessed=datetime.now()
            )
            
            self.cache[key] = entry
            self.current_memory += size_bytes
            
            # Evict if necessary
            self._evict_lru()
    
    def delete(self, key: str):
        """Delete value from memory cache"""
        with self.lock:
            if key in self.cache:
                entry = self.cache.pop(key)
                self.current_memory -= entry.size_bytes
    
    def exists(self, key: str) -> bool:
        """Check if key exists in memory cache"""
        with self.lock:
            return key in self.cache and not self.cache[key].is_expired()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory cache statistics"""
        with self.lock:
            total_access_count = sum(entry.access_count for entry in self.cache.values())
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "memory_usage_mb": self.current_memory / (1024 * 1024),
                "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
                "total_accesses": total_access_count,
                "utilization": len(self.cache) / self.max_size
            }
    
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.current_memory = 0


class HybridCache:
    """Hybrid cache using both memory and Redis"""
    
    def __init__(self, 
                 memory_cache: MemoryCache = None,
                 redis_cache: RedisCache = None,
                 default_ttl: int = 3600):
        self.memory_cache = memory_cache or MemoryCache()
        self.redis_cache = redis_cache or RedisCache()
        self.default_ttl = default_ttl
        
        # Cache statistics
        self.stats = {
            "hits": {"memory": 0, "redis": 0},
            "misses": 0,
            "sets": 0
        }
    
    def _generate_key(self, prefix: str, data: Any) -> str:
        """Generate cache key from data"""
        if isinstance(data, str):
            content = data
        elif isinstance(data, dict):
            content = json.dumps(data, sort_keys=True)
        else:
            content = str(data)
        
        key_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        return f"{prefix}:{key_hash}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (memory first, then Redis)"""
        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            self.stats["hits"]["memory"] += 1
            return value
        
        # Try Redis cache
        value = self.redis_cache.get(key)
        if value is not None:
            self.stats["hits"]["redis"] += 1
            # Store in memory cache for faster future access
            self.memory_cache.set(key, value, self.default_ttl)
            return value
        
        # Cache miss
        self.stats["misses"] += 1
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in both caches"""
        ttl = ttl or self.default_ttl
        
        # Set in both caches
        self.memory_cache.set(key, value, ttl)
        self.redis_cache.set(key, value, ttl)
        
        self.stats["sets"] += 1
    
    def delete(self, key: str):
        """Delete from both caches"""
        self.memory_cache.delete(key)
        self.redis_cache.delete(key)
    
    def exists(self, key: str) -> bool:
        """Check if key exists in either cache"""
        return self.memory_cache.exists(key) or self.redis_cache.exists(key)
    
    def get_hit_rate(self) -> float:
        """Calculate overall cache hit rate"""
        total_hits = self.stats["hits"]["memory"] + self.stats["hits"]["redis"]
        total_requests = total_hits + self.stats["misses"]
        
        if total_requests == 0:
            return 0.0
        
        return total_hits / total_requests
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        memory_stats = self.memory_cache.get_stats()
        redis_stats = self.redis_cache.get_stats()
        
        return {
            "hit_rate": self.get_hit_rate(),
            "memory_cache": memory_stats,
            "redis_cache": redis_stats,
            "access_stats": self.stats
        }


class EmbeddingCache:
    """Specialized cache for embeddings with similarity search"""
    
    def __init__(self, cache: HybridCache, similarity_threshold: float = 0.95):
        self.cache = cache
        self.similarity_threshold = similarity_threshold
        self.embedding_index: Dict[str, np.ndarray] = {}
    
    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    def get_similar_embedding(self, text: str, embedding: np.ndarray) -> Optional[Tuple[str, np.ndarray]]:
        """Find similar cached embedding"""
        best_similarity = 0.0
        best_key = None
        
        for key, cached_emb in self.embedding_index.items():
            similarity = self._compute_similarity(embedding, cached_emb)
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_key = key
        
        if best_key:
            cached_result = self.cache.get(best_key)
            if cached_result:
                return best_key, cached_result
        
        return None
    
    def cache_embedding(self, text: str, embedding: np.ndarray):
        """Cache embedding with similarity indexing"""
        key = self.cache._generate_key("embedding", text)
        
        # Store embedding
        self.cache.set(key, embedding)
        
        # Update similarity index
        self.embedding_index[key] = embedding
        
        # Limit index size to prevent memory issues
        if len(self.embedding_index) > 10000:
            # Remove oldest 20% of entries
            keys_to_remove = list(self.embedding_index.keys())[:2000]
            for k in keys_to_remove:
                del self.embedding_index[k]


def cached_function(cache: HybridCache, ttl: int = 3600, key_prefix: str = "func"):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_data = {
                "function": func.__name__,
                "args": args,
                "kwargs": kwargs
            }
            cache_key = cache._generate_key(key_prefix, cache_data)
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                logging.debug(f"Cache hit for {func.__name__}")
                return result
            
            # Execute function and cache result
            logging.debug(f"Cache miss for {func.__name__}, executing...")
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


class CacheManager:
    """Central cache management system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        
        # Initialize caches
        memory_config = config.get("memory", {})
        redis_config = config.get("redis", {})
        
        self.memory_cache = MemoryCache(
            max_size=memory_config.get("max_size", 1000),
            max_memory_mb=memory_config.get("max_memory_mb", 1024)
        )
        
        self.redis_cache = RedisCache(
            host=redis_config.get("host", "localhost"),
            port=redis_config.get("port", 6379),
            db=redis_config.get("db", 0),
            password=redis_config.get("password")
        )
        
        self.hybrid_cache = HybridCache(
            memory_cache=self.memory_cache,
            redis_cache=self.redis_cache,
            default_ttl=config.get("default_ttl", 3600)
        )
        
        self.embedding_cache = EmbeddingCache(
            cache=self.hybrid_cache,
            similarity_threshold=config.get("similarity_threshold", 0.95)
        )
        
        # Cache for different data types
        self.caches = {
            "general": self.hybrid_cache,
            "embeddings": self.embedding_cache,
            "memory": self.memory_cache,
            "redis": self.redis_cache
        }
        
        logging.info("Cache manager initialized")
    
    def get_cache(self, cache_type: str = "general") -> Any:
        """Get specific cache instance"""
        return self.caches.get(cache_type, self.hybrid_cache)
    
    def clear_all_caches(self):
        """Clear all caches"""
        self.memory_cache.clear()
        if self.redis_cache.available:
            try:
                self.redis_cache.redis_client.flushdb()
            except Exception as e:
                logging.error(f"Failed to clear Redis cache: {e}")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get statistics from all caches"""
        return {
            "hybrid_cache": self.hybrid_cache.get_stats(),
            "memory_cache": self.memory_cache.get_stats(),
            "redis_cache": self.redis_cache.get_stats()
        }


# Example usage
def main():
    """Example usage of caching system"""
    
    # Initialize cache manager
    cache_manager = CacheManager({
        "memory": {"max_size": 500, "max_memory_mb": 512},
        "redis": {"host": "localhost", "port": 6379},
        "default_ttl": 1800
    })
    
    # Get general cache
    cache = cache_manager.get_cache("general")
    
    # Example cached function
    @cached_function(cache, ttl=3600, key_prefix="expensive_computation")
    def expensive_computation(x: int, y: int) -> int:
        time.sleep(1)  # Simulate expensive operation
        return x * y + x ** y
    
    # Test caching
    print("First call (should be slow):")
    start = time.time()
    result1 = expensive_computation(5, 3)
    print(f"Result: {result1}, Time: {time.time() - start:.2f}s")
    
    print("\nSecond call (should be fast due to caching):")
    start = time.time()
    result2 = expensive_computation(5, 3)
    print(f"Result: {result2}, Time: {time.time() - start:.2f}s")
    
    # Get cache statistics
    stats = cache_manager.get_comprehensive_stats()
    print(f"\nCache statistics: {json.dumps(stats, indent=2, default=str)}")


if __name__ == "__main__":
    main()