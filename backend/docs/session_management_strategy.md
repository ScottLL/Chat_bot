# Session Management Strategy for Concurrent Request Handling

## 🎯 Objectives

Design a robust session management system that:
- **Isolates** user requests and state
- **Prevents** race conditions and data corruption
- **Optimizes** resource usage and performance
- **Scales** to handle multiple concurrent users
- **Maintains** consistency across async operations

---

## 🏗️ Current State Analysis

### **Issues with Current Implementation**

```python
# ❌ Current: Global shared agent
agent: DeFiQAAgent = None

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    global agent  # Shared across all requests
    result = agent.ask_question(request.question)
```

**Problems:**
1. **Shared State**: All users share the same agent instance
2. **No Isolation**: Requests can interfere with each other
3. **Thread Safety**: No protection against concurrent access
4. **Memory Leaks**: No cleanup of request-specific data

---

## 🛡️ Session Management Design

### **1. Session-Scoped Architecture**

#### **Session Data Model**
```python
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import uuid
import time

@dataclass
class SessionContext:
    """Session context for individual user requests."""
    session_id: str
    user_id: Optional[str] = None
    created_at: float = time.time()
    last_activity: float = time.time()
    
    # Request-specific state
    current_question: Optional[str] = None
    conversation_history: List[Dict[str, Any]] = None
    processing_stage: str = "initialized"
    
    # Performance tracking
    request_count: int = 0
    total_processing_time: float = 0.0
    
    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []
```

#### **Session Factory Pattern**
```python
from contextlib import asynccontextmanager
from typing import AsyncGenerator

class SessionManager:
    """Manages session lifecycle and cleanup."""
    
    def __init__(self):
        self._sessions: Dict[str, SessionContext] = {}
        self._lock = asyncio.Lock()
        self._cleanup_interval = 3600  # 1 hour
    
    @asynccontextmanager
    async def create_session(self, user_id: Optional[str] = None) -> AsyncGenerator[SessionContext, None]:
        """Create a new session with automatic cleanup."""
        session_id = str(uuid.uuid4())
        session = SessionContext(
            session_id=session_id,
            user_id=user_id
        )
        
        async with self._lock:
            self._sessions[session_id] = session
        
        try:
            yield session
        finally:
            await self._cleanup_session(session_id)
    
    async def _cleanup_session(self, session_id: str):
        """Clean up session resources."""
        async with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
```

### **2. Request-Scoped Agent Instances**

#### **Agent Factory with Dependency Injection**
```python
from fastapi import Depends
from typing import Annotated

class AgentFactory:
    """Factory for creating request-scoped agent instances."""
    
    def __init__(self):
        # Shared immutable resources
        self._dataset = None
        self._embedding_data = None
        self._embedding_service = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize shared resources once at startup."""
        if self._initialized:
            return
        
        print("🔄 Initializing shared agent resources...")
        self._dataset = QADataset().load_dataset()
        
        # Create thread-safe embedding service
        self._embedding_service = AsyncEmbeddingService()
        self._embedding_data = await self._embedding_service.compute_dataset_embeddings(
            self._dataset
        )
        
        self._initialized = True
        print("✅ Shared agent resources initialized")
    
    async def create_agent(self, session: SessionContext) -> 'AsyncDeFiQAAgent':
        """Create a request-scoped agent instance."""
        if not self._initialized:
            await self.initialize()
        
        return AsyncDeFiQAAgent(
            session_context=session,
            dataset=self._dataset,
            embedding_data=self._embedding_data,
            embedding_service=self._embedding_service
        )

# Global factory instance
agent_factory = AgentFactory()

async def get_session() -> SessionContext:
    """Dependency for creating request-scoped sessions."""
    async with SessionManager().create_session() as session:
        yield session

async def get_agent(session: SessionContext = Depends(get_session)) -> AsyncDeFiQAAgent:
    """Dependency for creating request-scoped agents."""
    return await agent_factory.create_agent(session)
```

### **3. Thread-Safe State Management**

#### **Async-Safe Caching**
```python
import asyncio
from typing import TypeVar, Generic

T = TypeVar('T')

class AsyncSafeCache(Generic[T]):
    """Thread-safe cache with async locks."""
    
    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, T] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._max_size = max_size
    
    async def get(self, key: str) -> Optional[T]:
        """Get value from cache."""
        async with self._lock:
            if key in self._cache:
                self._access_times[key] = time.time()
                return self._cache[key]
            return None
    
    async def set(self, key: str, value: T) -> None:
        """Set value in cache with LRU eviction."""
        async with self._lock:
            # Evict oldest if at capacity
            if len(self._cache) >= self._max_size and key not in self._cache:
                oldest_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
                del self._cache[oldest_key]
                del self._access_times[oldest_key]
            
            self._cache[key] = value
            self._access_times[key] = time.time()
    
    async def clear(self) -> None:
        """Clear all cached data."""
        async with self._lock:
            self._cache.clear()
            self._access_times.clear()
```

#### **Request Context Manager**
```python
from contextvars import ContextVar
from typing import Optional

# Context variables for request-scoped data
current_session: ContextVar[Optional[SessionContext]] = ContextVar('current_session', default=None)
current_agent: ContextVar[Optional['AsyncDeFiQAAgent']] = ContextVar('current_agent', default=None)

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
    def set_agent(agent: 'AsyncDeFiQAAgent'):
        """Set current agent in context."""
        current_agent.set(agent)
    
    @staticmethod
    def get_agent() -> Optional['AsyncDeFiQAAgent']:
        """Get current agent from context."""
        return current_agent.get()
```

### **4. Concurrent Processing Architecture**

#### **Rate-Limited Async Processing**
```python
import asyncio
from typing import List, Callable, Any

class ConcurrentProcessor:
    """Handles concurrent processing with rate limiting."""
    
    def __init__(self, max_concurrent: int = 10, rate_limit_per_second: int = 50):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._rate_limiter = asyncio.Semaphore(rate_limit_per_second)
        self._rate_reset_task = None
    
    async def start(self):
        """Start the rate limiter reset task."""
        self._rate_reset_task = asyncio.create_task(self._reset_rate_limiter())
    
    async def stop(self):
        """Stop the rate limiter reset task."""
        if self._rate_reset_task:
            self._rate_reset_task.cancel()
    
    async def _reset_rate_limiter(self):
        """Reset rate limiter every second."""
        while True:
            await asyncio.sleep(1)
            # Release all permits
            for _ in range(self._rate_limiter._value):
                self._rate_limiter.release()
    
    async def process_concurrent(
        self, 
        items: List[Any], 
        processor: Callable[[Any], Any]
    ) -> List[Any]:
        """Process items concurrently with rate limiting."""
        async def limited_processor(item):
            async with self._semaphore:  # Limit concurrency
                async with self._rate_limiter:  # Rate limiting
                    return await processor(item)
        
        tasks = [limited_processor(item) for item in items]
        return await asyncio.gather(*tasks, return_exceptions=True)
```

### **5. Updated FastAPI Integration**

#### **Async Endpoints with Session Management**
```python
from fastapi import FastAPI, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest,
    session: SessionContext = Depends(get_session),
    agent: AsyncDeFiQAAgent = Depends(get_agent),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Ask a question with session-scoped processing."""
    try:
        # Set request context
        RequestContextManager.set_session(session)
        RequestContextManager.set_agent(agent)
        
        # Update session state
        session.current_question = request.question
        session.request_count += 1
        session.last_activity = time.time()
        
        # Process question asynchronously
        start_time = time.time()
        result = await agent.ask_question_async(request.question)
        processing_time = time.time() - start_time
        
        # Update session metrics
        session.total_processing_time += processing_time
        session.processing_stage = result.get('processing_stage', 'completed')
        
        # Add conversation to history
        session.conversation_history.append({
            'question': request.question,
            'answer': result.get('response', ''),
            'confidence': result.get('similarity_scores', [0.0])[0] if result.get('similarity_scores') else 0.0,
            'processing_time': processing_time,
            'timestamp': time.time()
        })
        
        # Schedule background cleanup
        background_tasks.add_task(cleanup_old_conversations, session)
        
        return QuestionResponse(
            answer=result.get('response', ''),
            confidence=float(result.get('similarity_scores', [0.0])[0]) if result.get('similarity_scores') else 0.0,
            processing_stage=result.get('processing_stage', 'completed'),
            error=result.get('error')
        )
        
    except Exception as e:
        error = ErrorHandler.processing_error(str(e), session.processing_stage)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=format_error_for_response(error)
        )

async def cleanup_old_conversations(session: SessionContext):
    """Background task to clean up old conversation history."""
    max_history = 10
    if len(session.conversation_history) > max_history:
        session.conversation_history = session.conversation_history[-max_history:]
```

#### **Streaming with Session Context**
```python
@app.post("/ask-stream")
async def ask_question_stream(
    request: QuestionRequest,
    session: SessionContext = Depends(get_session),
    agent: AsyncDeFiQAAgent = Depends(get_agent)
):
    """Stream response with session management."""
    async def generate_stream():
        try:
            # Set context
            RequestContextManager.set_session(session)
            RequestContextManager.set_agent(agent)
            
            # Update session
            session.current_question = request.question
            session.last_activity = time.time()
            
            # Stream response
            async for chunk in agent.ask_question_stream_async(request.question):
                session.last_activity = time.time()
                yield f"data: {json.dumps(chunk)}\n\n"
                
        except Exception as e:
            error = ErrorHandler.processing_error(str(e), session.processing_stage)
            error_data = format_error_for_stream(error)
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )
```

---

## 🔧 Implementation Benefits

### **1. Concurrency Safety**
- **Session Isolation**: Each request has isolated state
- **Thread Safety**: Async locks protect shared resources
- **Race Condition Prevention**: Context variables and proper locking

### **2. Resource Management**
- **Memory Efficiency**: Automatic cleanup of session data
- **Connection Pooling**: Shared resources for immutable data
- **Graceful Cleanup**: Context managers ensure proper resource disposal

### **3. Performance Optimization**
- **Async Processing**: Non-blocking I/O operations
- **Rate Limiting**: Prevents API abuse and rate limit hits
- **Caching**: Multi-level caching with LRU eviction

### **4. Scalability**
- **Horizontal Scaling**: Session-scoped design supports multiple instances
- **Load Balancing**: Stateless request handling
- **Resource Sharing**: Efficient sharing of immutable resources

### **5. Monitoring and Debugging**
- **Session Tracking**: Detailed session metrics and history
- **Performance Metrics**: Processing time tracking
- **Error Context**: Session-aware error handling

---

## 📋 Implementation Steps

1. **Create Session Models** - Implement SessionContext and SessionManager
2. **Build Agent Factory** - Create request-scoped agent instances
3. **Add Thread-Safe Caching** - Implement AsyncSafeCache
4. **Update FastAPI Endpoints** - Add dependency injection
5. **Test Concurrent Load** - Verify isolation and performance

This session management strategy provides a robust foundation for handling concurrent requests while maintaining data integrity and optimal performance. 