"""
Session management for concurrent request handling.

This module provides session isolation, thread-safe state management,
and request-scoped resources for the DeFi Q&A application.
"""

import os
import time
import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, AsyncGenerator
from contextlib import asynccontextmanager
from contextvars import ContextVar

# Context variables for request-scoped data
current_session: ContextVar[Optional['SessionContext']] = ContextVar('current_session', default=None)
current_agent: ContextVar[Optional[Any]] = ContextVar('current_agent', default=None)


@dataclass
class SessionContext:
    """Session context for individual user requests."""
    session_id: str
    user_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    
    # Request-specific state
    current_question: Optional[str] = None
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    processing_stage: str = "initialized"
    
    # Performance tracking
    request_count: int = 0
    total_processing_time: float = 0.0
    
    def update_activity(self):
        """Update the last activity timestamp."""
        self.last_activity = time.time()
    
    def add_conversation(self, question: str, answer: str, confidence: float, processing_time: float):
        """Add a conversation entry to history."""
        self.conversation_history.append({
            'question': question,
            'answer': answer,
            'confidence': confidence,
            'processing_time': processing_time,
            'timestamp': time.time()
        })
        
        # Keep only last 10 conversations to manage memory
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'created_at': self.created_at,
            'last_activity': self.last_activity,
            'request_count': self.request_count,
            'total_processing_time': self.total_processing_time,
            'conversation_count': len(self.conversation_history),
            'processing_stage': self.processing_stage
        }


class SessionManager:
    """Manages session lifecycle and cleanup."""
    
    def __init__(self, cleanup_interval: int = 3600):  # 1 hour default
        self._sessions: Dict[str, SessionContext] = {}
        self._lock = asyncio.Lock()
        self._cleanup_interval = cleanup_interval
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """Start the session manager with cleanup task."""
        if self._running:
            return
        
        self._running = True
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        print("🔄 Session manager started with periodic cleanup")
    
    async def stop(self):
        """Stop the session manager and cleanup task."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Clean up all sessions
        async with self._lock:
            self._sessions.clear()
        
        print("🛑 Session manager stopped")
    
    @asynccontextmanager
    async def create_session(self, user_id: Optional[str] = None) -> AsyncGenerator[SessionContext, None]:
        """Create a new session with automatic cleanup."""
        session_id = str(uuid.uuid4())
        session = SessionContext(
            session_id=session_id,
            user_id=user_id
        )
        
        # Add session to manager
        async with self._lock:
            self._sessions[session_id] = session
        
        try:
            yield session
        finally:
            await self._cleanup_session(session_id)
    
    async def get_session(self, session_id: str) -> Optional[SessionContext]:
        """Get a session by ID."""
        async with self._lock:
            return self._sessions.get(session_id)
    
    async def _cleanup_session(self, session_id: str):
        """Clean up a specific session."""
        async with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
    
    async def _periodic_cleanup(self):
        """Periodically clean up old sessions."""
        while self._running:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_old_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in periodic cleanup: {e}")
    
    async def _cleanup_old_sessions(self):
        """Clean up sessions that haven't been active for too long."""
        current_time = time.time()
        expired_sessions = []
        
        async with self._lock:
            for session_id, session in self._sessions.items():
                # Clean up sessions inactive for more than 1 hour
                if current_time - session.last_activity > 3600:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del self._sessions[session_id]
        
        if expired_sessions:
            print(f"🧹 Cleaned up {len(expired_sessions)} expired sessions")
    
    async def get_active_sessions_count(self) -> int:
        """Get the number of active sessions."""
        async with self._lock:
            return len(self._sessions)
    
    async def get_session_stats(self) -> Dict[str, Any]:
        """Get overall session statistics."""
        async with self._lock:
            total_sessions = len(self._sessions)
            total_requests = sum(session.request_count for session in self._sessions.values())
            total_processing_time = sum(session.total_processing_time for session in self._sessions.values())
            
            return {
                'active_sessions': total_sessions,
                'total_requests': total_requests,
                'total_processing_time': total_processing_time,
                'average_processing_time': total_processing_time / max(total_requests, 1),
                'cleanup_interval': self._cleanup_interval
            }


class RequestContextManager:
    """Manages request-scoped context variables."""
    
    @staticmethod
    def set_session(session: SessionContext):
        """Set current session in context."""
        current_session.set(session)
    
    @staticmethod
    def get_session() -> Optional[SessionContext]:
        """Get current session from context."""
        return current_session.get()
    
    @staticmethod
    def set_agent(agent: Any):
        """Set current agent in context."""
        current_agent.set(agent)
    
    @staticmethod
    def get_agent() -> Optional[Any]:
        """Get current agent from context."""
        return current_agent.get()
    
    @staticmethod
    def clear_context():
        """Clear all context variables."""
        current_session.set(None)
        current_agent.set(None)


class AsyncSafeCache:
    """Thread-safe cache with async locks and LRU eviction."""
    
    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, Any] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._max_size = max_size
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self._lock:
            if key in self._cache:
                self._access_times[key] = time.time()
                return self._cache[key]
            return None
    
    async def set(self, key: str, value: Any) -> None:
        """Set value in cache with LRU eviction."""
        async with self._lock:
            # Evict oldest if at capacity
            if len(self._cache) >= self._max_size and key not in self._cache:
                oldest_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
                del self._cache[oldest_key]
                del self._access_times[oldest_key]
            
            self._cache[key] = value
            self._access_times[key] = time.time()
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                del self._access_times[key]
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all cached data."""
        async with self._lock:
            self._cache.clear()
            self._access_times.clear()
    
    async def size(self) -> int:
        """Get current cache size."""
        async with self._lock:
            return len(self._cache)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            return {
                'size': len(self._cache),
                'max_size': self._max_size,
                'utilization': len(self._cache) / self._max_size if self._max_size > 0 else 0
            }


class ConcurrentProcessor:
    """Handles concurrent processing with rate limiting."""
    
    def __init__(self, max_concurrent: int = 10, rate_limit_per_second: int = 50):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._rate_limit = rate_limit_per_second
        self._rate_tokens = rate_limit_per_second
        self._rate_lock = asyncio.Lock()
        self._rate_reset_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """Start the concurrent processor with rate limiting."""
        if self._running:
            return
        
        self._running = True
        self._rate_reset_task = asyncio.create_task(self._reset_rate_tokens())
        print(f"🚀 Concurrent processor started (max_concurrent={self._semaphore._value}, rate_limit={self._rate_limit}/s)")
    
    async def stop(self):
        """Stop the concurrent processor."""
        self._running = False
        if self._rate_reset_task:
            self._rate_reset_task.cancel()
            try:
                await self._rate_reset_task
            except asyncio.CancelledError:
                pass
        print("🛑 Concurrent processor stopped")
    
    async def _reset_rate_tokens(self):
        """Reset rate limiting tokens every second."""
        while self._running:
            try:
                await asyncio.sleep(1)
                async with self._rate_lock:
                    self._rate_tokens = self._rate_limit
            except asyncio.CancelledError:
                break
    
    async def _acquire_rate_token(self):
        """Acquire a rate limiting token."""
        while True:
            async with self._rate_lock:
                if self._rate_tokens > 0:
                    self._rate_tokens -= 1
                    return
            
            # Wait a bit and try again
            await asyncio.sleep(0.01)
    
    async def process_with_limits(self, coro):
        """Process a coroutine with concurrency and rate limiting."""
        await self._acquire_rate_token()
        async with self._semaphore:
            return await coro
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        async with self._rate_lock:
            return {
                'max_concurrent': self._semaphore._value,
                'available_slots': self._semaphore._value,
                'rate_limit': self._rate_limit,
                'available_tokens': self._rate_tokens
            }


# Global instances
session_manager = SessionManager()
request_cache = AsyncSafeCache(max_size=1000)
processor = ConcurrentProcessor(max_concurrent=10, rate_limit_per_second=50)


async def initialize_session_management():
    """Initialize the session management system."""
    await session_manager.start()
    await processor.start()
    print("✅ Session management system initialized")


async def shutdown_session_management():
    """Shutdown the session management system."""
    await session_manager.stop()
    await processor.stop()
    await request_cache.clear()
    print("✅ Session management system shutdown complete")


async def test_session_management():
    """Test the session management system."""
    print("🧪 Testing Session Management System...")
    
    try:
        # Initialize system
        await initialize_session_management()
        
        # Test session creation
        async with session_manager.create_session(user_id="test_user") as session:
            print(f"✅ Session created: {session.session_id}")
            
            # Test context management
            RequestContextManager.set_session(session)
            retrieved_session = RequestContextManager.get_session()
            assert retrieved_session == session
            print("✅ Request context management working")
            
            # Test session updates
            session.add_conversation("Test question", "Test answer", 0.8, 1.5)
            stats = session.get_stats()
            print(f"✅ Session stats: {stats}")
        
        # Test cache
        await request_cache.set("test_key", "test_value")
        value = await request_cache.get("test_key")
        assert value == "test_value"
        print("✅ Async safe cache working")
        
        # Test processor
        async def test_task():
            await asyncio.sleep(0.1)
            return "processed"
        
        result = await processor.process_with_limits(test_task())
        assert result == "processed"
        print("✅ Concurrent processor working")
        
        # Get system stats
        session_stats = await session_manager.get_session_stats()
        cache_stats = await request_cache.get_stats()
        processor_stats = await processor.get_stats()
        
        print(f"✅ System stats:")
        print(f"   Sessions: {session_stats}")
        print(f"   Cache: {cache_stats}")
        print(f"   Processor: {processor_stats}")
        
        # Shutdown
        await shutdown_session_management()
        print("✅ Session management test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_session_management()) 