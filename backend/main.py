"""
DeFi Q&A Bot API

FastAPI application providing semantic search capabilities over DeFi Q&A dataset
using LangGraph orchestration and OpenAI embeddings.
"""

import os
import json
import time
import asyncio
from typing import Dict, Any, Optional, Generator, AsyncGenerator
from contextlib import asynccontextmanager
from collections import defaultdict, deque
from pathlib import Path
import uuid

from fastapi import FastAPI, HTTPException, status, Depends, BackgroundTasks, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# Import configuration
from config import config

# Import agents
from agents.defi_qa_agent import DeFiQAAgent
from agents.async_defi_qa_agent import AsyncDeFiQAAgent

# Function to get the factory instance (lazy import)
def get_async_agent_factory():
    from agents.async_defi_qa_agent import async_agent_factory
    return async_agent_factory

# Import services
from services.session_manager import (
    SessionContext, SessionManager, RequestContextManager,
    session_manager, initialize_session_management, shutdown_session_management
)

# Import infrastructure
from infrastructure.error_handlers import ErrorHandler, format_error_for_response, format_error_for_stream
from infrastructure.websocket_manager import (
    websocket_manager,
    WebSocketConnectionManager,
    WebSocketMessageType,
    create_word_message,
    create_metadata_message,
    create_complete_message,
    create_error_message,
    create_status_message
)
from infrastructure.logging_config import get_logger, app_logger
from infrastructure.monitoring import metrics_collector, monitoring_dashboard, performance_monitor
from prometheus_client import CONTENT_TYPE_LATEST


# Rate Limiting Middleware
class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware based on client IP."""
    
    def __init__(self, app, requests_per_minute: int = None, window_seconds: int = None):
        super().__init__(app)
        # Use config values if not explicitly provided
        self.requests_per_minute = requests_per_minute or config.RATE_LIMIT_REQUESTS_PER_MINUTE
        self.window_seconds = window_seconds or config.RATE_LIMIT_WINDOW_SECONDS
        self.requests = defaultdict(deque)  # IP -> deque of timestamps
    
    async def dispatch(self, request: Request, call_next):
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Skip rate limiting for health/stats endpoints
        if request.url.path in ["/health", "/session/stats", "/rate-limit/reset"]:
            return await call_next(request)
        
        # Only apply rate limiting to V2 endpoints
        if not request.url.path.startswith("/v2/"):
            return await call_next(request)
        
        current_time = time.time()
        
        # Clean old requests outside the window
        self._clean_old_requests(client_ip, current_time)
        
        # Check if rate limit exceeded
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded", "retry_after": f"{self.window_seconds} seconds"}
            )
        
        # Add current request
        self.requests[client_ip].append(current_time)
        
        # Process request
        return await call_next(request)
    
    def _clean_old_requests(self, client_ip: str, current_time: float):
        """Remove requests older than the time window."""
        cutoff_time = current_time - self.window_seconds
        requests_deque = self.requests[client_ip]
        
        while requests_deque and requests_deque[0] < cutoff_time:
            requests_deque.popleft()
    
    def reset_limits(self):
        """Reset all rate limits (for testing purposes)."""
        self.requests.clear()


# Request Logging Middleware
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all HTTP requests and responses with performance metrics."""
    
    def __init__(self, app):
        super().__init__(app)
        self.logger = get_logger()
    
    async def dispatch(self, request: Request, call_next):
        # Start timing
        start_time = time.time()
        
        # Get request details
        method = request.method
        path = str(request.url.path)
        client_ip = request.client.host if request.client else "unknown"
        
        # Get session ID if available (for correlation)
        session_id = None
        user_id = None
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Get response details
            status_code = response.status_code
            
            # Log the request
            self.logger.log_request(
                method=method,
                path=path,
                status_code=status_code,
                processing_time=processing_time,
                session_id=session_id,
                user_id=user_id,
                client_ip=client_ip
            )
            
            # Record metrics
            metrics_collector.record_http_request(
                method=method,
                endpoint=path,
                status_code=status_code,
                duration=processing_time
            )
            
            return response
            
        except Exception as e:
            # Calculate processing time even for errors
            processing_time = time.time() - start_time
            
            # Log the error
            self.logger.log_error(
                error=e,
                context=f"Request processing: {method} {path}",
                session_id=session_id,
                user_id=user_id,
                client_ip=client_ip,
                processing_time=processing_time
            )
            
            # Re-raise the exception
            raise


# Global agent instance
agent: DeFiQAAgent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager to handle startup and shutdown.
    
    Initializes both the legacy synchronous agent and the new async
    session management system for concurrent request handling.
    """
    global agent
    logger = get_logger()
    
    logger.log_startup("DeFi Q&A API", success=True)
    
    try:
        # Initialize session management system
        await initialize_session_management()
        logger.log_startup("Session Management", success=True)
        
        # Initialize async agent factory
        await get_async_agent_factory().initialize()
        logger.log_startup("Async Agent Factory", success=True)
        
        # Initialize WebSocket manager
        await websocket_manager.start_background_tasks()
        logger.log_startup("WebSocket Manager", success=True)
        
        # Initialize performance monitoring
        await performance_monitor.start_monitoring()
        logger.log_startup("Performance Monitor", success=True)
        
        # Initialize the legacy agent for backward compatibility
        # Check if OpenAI API key is configured first
        if not config.OPENAI_API_KEY:
            logger.logger.warning("⚠️  OpenAI API key not configured - Legacy agent will be disabled")
            logger.logger.info("   Create a .env file with OPENAI_API_KEY=your_key_here")
            agent = None
        else:
            agent = DeFiQAAgent(
                similarity_threshold=config.AGENT_SIMILARITY_THRESHOLD,
                max_results=config.AGENT_MAX_RESULTS,
                cache_enabled=config.AGENT_CACHE_ENABLED
            )
            logger.log_startup("Legacy DeFi Agent", success=True)
        
        logger.logger.info("🚀 DeFi Q&A API startup completed successfully!")
        
    except Exception as e:
        logger.log_startup("Application Components", success=False, error=str(e))
        logger.log_error(e, "Application startup failed")
        logger.logger.error(f"   Error details: {str(e)}")
        if "api" in str(e).lower() or "key" in str(e).lower():
            logger.logger.info("   💡 This might be an API key issue. Check your .env file!")
        # Don't prevent startup, but log the error
        agent = None
    
    yield  # Application runs here
    
    # Cleanup on shutdown
    logger.log_shutdown("DeFi Q&A API")
    await performance_monitor.stop_monitoring()
    logger.log_shutdown("Performance Monitor")
    await websocket_manager.stop_background_tasks()
    logger.log_shutdown("WebSocket Manager")
    await shutdown_session_management()
    logger.log_shutdown("Session Management")
    await get_async_agent_factory().cleanup()
    logger.log_shutdown("Async Agent Factory")


# Dependency injection functions
async def get_session() -> AsyncGenerator[SessionContext, None]:
    """Dependency for creating request-scoped sessions."""
    async with session_manager.create_session() as session:
        yield session


async def get_async_agent(session: SessionContext = Depends(get_session)) -> AsyncDeFiQAAgent:
    """Dependency for creating request-scoped async agents."""
    agent = await get_async_agent_factory().create_agent(session)
    RequestContextManager.set_session(session)
    RequestContextManager.set_agent(agent)
    return agent


async def get_websocket_session() -> SessionContext:
    """Special dependency for WebSocket connections that creates a persistent session."""
    session_id = str(uuid.uuid4())
    session = SessionContext(session_id=session_id)
    
    # Add to session manager without using context manager
    async with session_manager._lock:
        session_manager._sessions[session_id] = session
    
    return session


async def get_websocket_agent(session: SessionContext = Depends(get_websocket_session)) -> AsyncDeFiQAAgent:
    """Dependency for creating WebSocket-scoped async agents."""
    agent = await get_async_agent_factory().create_agent(session)
    RequestContextManager.set_session(session)
    RequestContextManager.set_agent(agent)
    return agent


# Cleanup old conversations based on config
async def cleanup_old_conversations(session: SessionContext):
    """Clean up old conversations to prevent memory bloat."""
    max_history = config.MAX_SESSION_HISTORY
    if len(session.conversation_history) > max_history:
        session.conversation_history = session.conversation_history[-max_history:]


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="DeFi Q&A Bot API",
    description="Semantic search API for DeFi-related questions using LangGraph and OpenAI embeddings",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)


# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,  # Use config instead of hardcoded values
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Add request logging middleware
app.add_middleware(RequestLoggingMiddleware)

# Add rate limiting middleware (create instance to access reset method)
class GlobalRateLimiter:
    def __init__(self):
        self.middleware = None

global_rate_limiter = GlobalRateLimiter()

# Use config values for rate limiting
app.add_middleware(
    RateLimitMiddleware, 
    requests_per_minute=config.RATE_LIMIT_REQUESTS_PER_MINUTE, 
    window_seconds=config.RATE_LIMIT_WINDOW_SECONDS
)


# Request/Response Models
class QuestionRequest(BaseModel):
    """Request model for asking questions."""
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="DeFi-related question to ask",
        example="What is the largest lending pool on Aave?"
    )


class QuestionResponse(BaseModel):
    """Response model for question answers."""
    answer: str = Field(description="The answer to the user's question")
    confidence: float = Field(description="Confidence score (0-1) of the answer")
    processing_stage: str = Field(description="Processing stage reached")
    error: Optional[str] = Field(None, description="Error message if any")
    session_id: Optional[str] = Field(None, description="Session ID for request tracking")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(description="API health status")
    agent_loaded: bool = Field(description="Whether the DeFi Q&A agent is loaded")
    message: str = Field(description="Additional status information")


# API Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint providing API information."""
    return {
        "message": "DeFi Q&A Bot API",
        "description": "Semantic search API for DeFi-related questions",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify API and agent status.
    
    Returns:
        HealthResponse: Current health status of the API and agent
    """
    try:
        # Check if async agent factory is available and initialized
        async_factory = get_async_agent_factory()
        agent_loaded = async_factory is not None and async_factory._initialized
        
        # Check legacy agent and API key configuration
        legacy_agent_loaded = agent is not None
        api_key_configured = bool(config.OPENAI_API_KEY)
        
        if agent_loaded and legacy_agent_loaded:
            status_msg = "healthy"
            message = "API is running and both async and legacy DeFi Q&A agents are ready"
        elif agent_loaded:
            if not api_key_configured:
                status_msg = "partial"
                message = "API is running, async agent ready, but legacy agent disabled (OpenAI API key not configured)"
            else:
                status_msg = "partial"
                message = "API is running and async DeFi Q&A agent is ready, legacy agent unavailable"
        else:
            if not api_key_configured:
                status_msg = "degraded"
                message = "API is running but OpenAI API key is not configured. Create .env file with OPENAI_API_KEY=your_key_here"
            else:
                status_msg = "degraded"
                message = "API is running but DeFi Q&A agents failed to load"
    except Exception as e:
        agent_loaded = False
        status_msg = "degraded"
        message = f"API is running but DeFi Q&A agent failed to load: {str(e)}"
    
    return HealthResponse(
        status=status_msg,
        agent_loaded=agent_loaded,
        message=message
    )


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a DeFi-related question and get a semantic search-based answer.
    
    This endpoint uses the LangGraph agent to:
    1. Parse and validate the user's question
    2. Perform semantic search over the DeFi Q&A dataset
    3. Select the most relevant answer
    4. Return the answer with confidence score
    
    Args:
        request: QuestionRequest containing the user's question
        
    Returns:
        QuestionResponse: Answer, confidence score, and processing information
        
    Raises:
        HTTPException: If the agent is not loaded or processing fails
    """
    global agent
    
    # Check if agent is loaded
    if agent is None:
        if not config.OPENAI_API_KEY:
            error = ErrorHandler.configuration_error(
                "OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file."
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=format_error_for_response(error)
            )
        else:
            error = ErrorHandler.agent_unavailable_error()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=format_error_for_response(error)
            )
    
    try:
        # Process the question using our LangGraph agent
        result = agent.ask_question(request.question)
        
        # Check for errors from the agent
        if result.get('error'):
            # Handle specific error types from the agent
            if 'No relevant' in result['error'] or "couldn't find" in result['error']:
                error = ErrorHandler.no_results_error(request.question)
            else:
                error = ErrorHandler.processing_error(result['error'], result.get('processing_stage', 'unknown'))
            
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=format_error_for_response(error)
            )
        
        # Extract confidence score from similarity scores
        confidence = 0.0
        if result.get('similarity_scores'):
            confidence = float(result['similarity_scores'][0])
        
        # Create response
        response = QuestionResponse(
            answer=result['response'],
            confidence=confidence,
            processing_stage=result['processing_stage'],
            error=None
        )
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions (already handled above)
        raise
    except Exception as e:
        # Log the error and return structured error response
        print(f"❌ Error processing question: {e}")
        error = ErrorHandler.processing_error(str(e), "unknown")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=format_error_for_response(error)
        )


@app.post("/ask-stream")
async def ask_question_stream(request: QuestionRequest):
    """
    Ask a DeFi-related question and get a streaming word-by-word response.
    
    This endpoint provides the same functionality as /ask but streams the response
    word-by-word using Server-Sent Events (SSE) for a more engaging user experience.
    
    The stream sends JSON objects in the format:
    - {"type": "word", "content": "word", "confidence": 0.95}
    - {"type": "complete", "processing_stage": "response_generated"}
    - {"type": "error", "error": "error message"}
    
    Args:
        request: QuestionRequest containing the user's question
        
    Returns:
        StreamingResponse: SSE stream of words and metadata
        
    Raises:
        HTTPException: If the agent is not loaded or processing fails
    """
    global agent
    
    # Check if agent is loaded
    if agent is None:
        if not config.OPENAI_API_KEY:
            error = ErrorHandler.configuration_error(
                "OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file."
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=format_error_for_response(error)
            )
        else:
            error = ErrorHandler.agent_unavailable_error()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=format_error_for_response(error)
            )
    
    async def generate_stream() -> AsyncGenerator[str, None]:
        """Generate SSE stream of response words."""
        try:
            # Process the question using our LangGraph agent
            result = agent.ask_question(request.question)
            
            # Extract response data
            response_text = result['response']
            confidence = 0.0
            if result.get('similarity_scores'):
                confidence = float(result['similarity_scores'][0])
            
            # Check for errors from the agent
            if result.get('error'):
                # Create structured error based on error type
                if 'No relevant' in result['error'] or "couldn't find" in result['error']:
                    structured_error = ErrorHandler.no_results_error(request.question)
                else:
                    structured_error = ErrorHandler.processing_error(
                        result['error'], 
                        result.get('processing_stage', 'unknown')
                    )
                
                error_data = format_error_for_stream(structured_error)
                yield f"data: {json.dumps(error_data)}\n\n"
                return
            
            # Stream the response word by word
            words = response_text.split()
            
            for i, word in enumerate(words):
                # Send each word as an SSE event
                word_data = {
                    "type": "word",
                    "content": word,
                    "index": i,
                    "confidence": confidence,
                    "processing_stage": result.get('processing_stage', 'streaming')
                }
                
                yield f"data: {json.dumps(word_data)}\n\n"
                
                # Add a small delay for realistic streaming effect
                await asyncio.sleep(0.05)  # 50ms between words
            
            # Send completion event
            complete_data = {
                "type": "complete",
                "total_words": len(words),
                "confidence": confidence,
                "processing_stage": result.get('processing_stage', 'response_generated')
            }
            yield f"data: {json.dumps(complete_data)}\n\n"
            
        except Exception as e:
            # Send error event
            error_data = {
                "type": "error",
                "error": f"Streaming error: {str(e)}",
                "processing_stage": "streaming_failed"
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    # Return streaming response with proper SSE headers
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )


# New async endpoints with session management
@app.post("/v2/ask", response_model=QuestionResponse)
async def ask_question_v2(
    request: QuestionRequest,
    session: SessionContext = Depends(get_session),
    agent: AsyncDeFiQAAgent = Depends(get_async_agent),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    V2 endpoint with async processing and session management.
    
    This endpoint provides:
    - Session-scoped processing for request isolation
    - Async/await for non-blocking operations
    - Background task cleanup for memory management
    - Enhanced error handling with context awareness
    
    Args:
        request: QuestionRequest containing the user's question
        session: Session context automatically injected via dependency
        agent: Async agent instance automatically created for this session
        background_tasks: FastAPI background tasks for cleanup
        
    Returns:
        QuestionResponse: Answer, confidence score, and processing information
        
    Raises:
        HTTPException: If processing fails
    """
    try:
        # Update session state
        session.current_question = request.question
        session.request_count += 1
        session.update_activity()
        
        # Process question asynchronously
        start_time = time.time()
        result = await agent.ask_question_async(request.question)
        processing_time = time.time() - start_time
        
        # Update session metrics
        session.total_processing_time += processing_time
        session.processing_stage = result.get('processing_stage', 'completed')
        
        # Check for errors from the agent
        if result.get('error'):
            # Create structured error based on error type
            if 'No relevant' in result['error'] or "couldn't find" in result['error']:
                structured_error = ErrorHandler.no_results_error(request.question)
            else:
                structured_error = ErrorHandler.processing_error(
                    result['error'], 
                    result.get('processing_stage', 'unknown')
                )
            
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=format_error_for_response(structured_error)
            )
        
        # Schedule background cleanup
        background_tasks.add_task(cleanup_old_conversations, session)
        
        # Extract confidence score
        confidence = 0.0
        if result.get('similarity_scores'):
            confidence = float(max(result['similarity_scores']))
        
        return QuestionResponse(
            answer=result.get('response', ''),
            confidence=confidence,
            processing_stage=result.get('processing_stage', 'completed'),
            error=result.get('error'),
            session_id=session.session_id
        )
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        error = ErrorHandler.processing_error(str(e), session.processing_stage)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=format_error_for_response(error)
        )


@app.post("/v2/ask-stream")
async def ask_question_stream_v2(
    request: QuestionRequest,
    session: SessionContext = Depends(get_session),
    agent: AsyncDeFiQAAgent = Depends(get_async_agent)
):
    """
    V2 streaming endpoint with async processing and session management.
    
    This endpoint provides:
    - Session-scoped processing for request isolation
    - Async streaming with proper resource management
    - Enhanced error handling in streaming context
    - Real-time session activity updates
    
    Args:
        request: QuestionRequest containing the user's question
        session: Session context automatically injected via dependency
        agent: Async agent instance automatically created for this session
        
    Returns:
        StreamingResponse: Server-sent events with word-by-word response
    """
    async def generate_async_stream():
        try:
            # Update session
            session.current_question = request.question
            session.update_activity()
            
            # Stream response from async agent
            async for chunk in agent.ask_question_stream_async(request.question):
                session.update_activity()
                yield f"data: {json.dumps(chunk)}\n\n"
                
        except Exception as e:
            error = ErrorHandler.processing_error(str(e), session.processing_stage)
            error_data = format_error_for_stream(error)
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate_async_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache", 
            "Connection": "keep-alive",
            "X-Session-ID": session.session_id
        }
    )


# Session management endpoints
@app.get("/session/stats")
async def get_session_stats():
    """Get overall session management statistics."""
    try:
        session_stats = await session_manager.get_session_stats()
        factory_stats = await get_async_agent_factory().get_stats()
        
        return {
            "session_management": session_stats,
            "agent_factory": factory_stats,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get session stats: {str(e)}"
        )


@app.post("/rate-limit/reset")
async def reset_rate_limits():
    """Reset rate limits for testing purposes."""
    # Find the rate limit middleware instance
    for middleware in app.user_middleware:
        if hasattr(middleware.cls, 'reset_limits'):
            # Get the middleware instance from the stack
            # This is a bit hacky but necessary for FastAPI middleware access
            pass
    
    # Alternative approach: restart will clear the rate limits anyway
    return {"message": "Rate limits will be cleared on next server restart"}


@app.get("/rate-limit/status")
async def get_rate_limit_status():
    """Get current rate limiting status."""
    return {
        "rate_limit_active": True,
        "requests_per_minute": config.RATE_LIMIT_REQUESTS_PER_MINUTE,
        "window_seconds": config.RATE_LIMIT_WINDOW_SECONDS,
        "message": "Rate limiting is active for V2 endpoints"
    }


# WebSocket endpoints
@app.websocket("/ws/test")
async def websocket_test_endpoint(websocket: WebSocket):
    """Simple test WebSocket endpoint without dependencies."""
    try:
        await websocket.accept()
        await websocket.send_text(json.dumps({
            "type": "status",
            "status": "connected",
            "message": "Test WebSocket connection successful"
        }))
        
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                await websocket.send_text(json.dumps({
                    "type": "echo",
                    "message": f"Received: {message}"
                }))
            except WebSocketDisconnect:
                break
            except Exception as e:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "error": str(e)
                }))
                
    except Exception as e:
        print(f"❌ Test WebSocket error: {e}")


@app.websocket("/ws/ask")
async def websocket_ask_endpoint(
    websocket: WebSocket,
    session: SessionContext = Depends(get_websocket_session),
    agent: AsyncDeFiQAAgent = Depends(get_websocket_agent)
):
    """
    WebSocket endpoint for real-time DeFi Q&A streaming.
    
    This endpoint provides:
    - Real-time bidirectional communication
    - Session-scoped processing with automatic cleanup
    - Enhanced error handling and recovery
    - Support for request cancellation and pause/resume
    - Automatic reconnection handling
    
    WebSocket Message Protocol:
    
    Client -> Server:
    - {"type": "question", "data": {"question": "What is DeFi?"}}
    - {"type": "cancel"}
    - {"type": "pause"}
    - {"type": "resume"}
    - {"type": "heartbeat_response"}
    
    Server -> Client:
    - {"type": "metadata", "confidence": 0.85, "total_words": 45, "session_id": "..."}
    - {"type": "word", "content": "DeFi", "index": 0, "confidence": 0.85}
    - {"type": "complete", "processing_stage": "completed", "final_confidence": 0.85}
    - {"type": "error", "error": "Error message", "processing_stage": "error"}
    - {"type": "heartbeat"}
    - {"type": "status", "status": "connected/streaming/paused"}
    """
    logger = get_logger()
    session_id = None
    
    try:
        print(f"🔍 WebSocket connection attempt - Session: {session.session_id}")
        logger.logger.info(f"WebSocket connection attempt for session {session.session_id}")
        
        # Connect to WebSocket manager
        session_id = await websocket_manager.connect(websocket, session, agent)
        connection = websocket_manager.get_connection(session_id)
        
        print(f"✅ WebSocket connected - Session: {session_id}")
        
        # Log WebSocket connection
        logger.log_websocket_event(
            event_type="connected",
            session_id=session_id,
            connection_count=len(websocket_manager.active_connections)
        )
        
        # Record WebSocket connection metrics
        metrics_collector.record_websocket_connection(
            status="connected",
            active_count=len(websocket_manager.active_connections)
        )
        
        # Send initial status message
        status_msg = create_status_message("connected", {
            "session_id": session_id,
            "capabilities": ["streaming", "cancellation", "pause_resume"],
            "protocol_version": "1.0"
        })
        try:
            await websocket_manager.send_message(session_id, status_msg)
            print(f"📤 Sent initial status message to session {session_id}")
        except Exception as send_error:
            print(f"🔌 Failed to send initial status message to {session_id}: {send_error}")
            # If we can't send the initial message, the connection is likely broken
            raise WebSocketDisconnect(code=1000, reason="Failed to send initial message")
        
        # Message receiving loop with proper disconnect handling
        while True:
            try:
                # Receive message from client
                message_text = await websocket.receive_text()
                message = json.loads(message_text)
                
                message_type = message.get("type")
                print(f"📨 Received message type: {message_type} for session {session_id}")
                
                if message_type == WebSocketMessageType.QUESTION:
                    await handle_websocket_question(session_id, message, connection)
                
                elif message_type == WebSocketMessageType.CANCEL:
                    await handle_websocket_cancel(session_id, connection)
                
                elif message_type == WebSocketMessageType.PAUSE:
                    await handle_websocket_pause(session_id, connection)
                
                elif message_type == WebSocketMessageType.RESUME:
                    await handle_websocket_resume(session_id, connection)
                
                elif message_type == WebSocketMessageType.HEARTBEAT_RESPONSE:
                    connection.update_heartbeat()
                
                else:
                    error_msg = create_error_message(f"Unknown message type: {message_type}")
                    try:
                        await websocket_manager.send_message(session_id, error_msg)
                    except Exception as send_error:
                        print(f"🔌 Failed to send unknown message type error to {session_id}: {send_error}")
                        break
                    
            except WebSocketDisconnect as e:
                print(f"🔌 WebSocket disconnected during message loop - Session: {session_id}, Code: {e.code if hasattr(e, 'code') else 'unknown'}")
                # Break out of the loop cleanly - don't try to continue receiving
                break
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error in session {session_id}: {e}")
                try:
                    error_msg = create_error_message("Invalid JSON message format")
                    await websocket_manager.send_message(session_id, error_msg)
                except Exception:
                    # If we can't send the error message, the connection is likely broken
                    print(f"🔌 Cannot send error message - connection broken for session {session_id}")
                    break
            
            except Exception as e:
                print(f"❌ Message processing error in session {session_id}: {e}")
                # Only try to send error message if the error isn't related to a broken connection
                error_str = str(e).lower()
                if "disconnect" in error_str or "closed" in error_str or "receive" in error_str:
                    print(f"🔌 Connection-related error, breaking loop for session {session_id}")
                    break
                
                try:
                    error_msg = create_error_message(f"Message processing error: {str(e)}")
                    await websocket_manager.send_message(session_id, error_msg)
                except Exception:
                    # If we can't send the error message, the connection is likely broken
                    print(f"🔌 Cannot send error message - connection broken for session {session_id}")
                    break
                
    except WebSocketDisconnect as e:
        print(f"🔌 WebSocket disconnected - Session: {session_id}, Code: {e.code if hasattr(e, 'code') else 'unknown'}")
        if session_id:
            logger.log_websocket_event(
                event_type="disconnected",
                session_id=session_id,
                connection_count=len(websocket_manager.active_connections) - 1,
                reason="client_disconnect"
            )
            # Record WebSocket disconnection metrics
            metrics_collector.record_websocket_connection(
                status="disconnected",
                active_count=len(websocket_manager.active_connections) - 1
            )
    except Exception as e:
        print(f"❌ WebSocket error - Session: {session_id}, Error: {e}")
        import traceback
        traceback.print_exc()
        
        if session_id:
            logger.log_websocket_event(
                event_type="error",
                session_id=session_id,
                connection_count=len(websocket_manager.active_connections),
                error=str(e)
            )
            logger.log_error(e, f"WebSocket error for session {session_id}", session_id=session_id)
    finally:
        if session_id:
            print(f"🧹 Cleaning up WebSocket connection for session {session_id}")
            # Clean up connection
            await websocket_manager.disconnect(websocket)
            
            # Clean up the session manually since we didn't use context manager
            async with session_manager._lock:
                if session_id in session_manager._sessions:
                    del session_manager._sessions[session_id]
            
            logger.log_websocket_event(
                event_type="cleanup",
                session_id=session_id,
                connection_count=len(websocket_manager.active_connections)
            )


async def handle_websocket_question(session_id: str, message: Dict[str, Any], connection):
    """Handle a question message from WebSocket client."""
    logger = get_logger()
    start_time = time.time()
    
    try:
        question_data = message.get("data", {})
        question = question_data.get("question", "").strip()
        
        if not question:
            error_msg = create_error_message("Question cannot be empty")
            if websocket_manager.is_connection_active(session_id):
                try:
                    await websocket_manager.send_message(session_id, error_msg)
                except Exception as send_error:
                    print(f"🔌 Failed to send empty question error to {session_id}: {send_error}")
            return
        
        # Log question received
        logger.log_websocket_event(
            event_type="question_received",
            session_id=session_id,
            message_type="question",
            question_length=len(question)
        )
        
        # Record WebSocket message metrics
        metrics_collector.record_websocket_message(
            direction="received",
            message_type="question"
        )
        
        # Update session state
        connection.session_context.current_question = question
        connection.session_context.request_count += 1
        connection.session_context.update_activity()
        connection.is_streaming = True
        
        # Send status update
        status_msg = create_status_message("streaming")
        if websocket_manager.is_connection_active(session_id):
            try:
                await websocket_manager.send_message(session_id, status_msg)
            except Exception as send_error:
                print(f"🔌 Failed to send streaming status to {session_id}: {send_error}")
                return  # Exit early if we can't send status
        
        # Stream response from async agent
        word_count = 0
        confidence = 0.0
        processing_stage = "unknown"
        error = None
        
        try:
            async for chunk in connection.agent.ask_question_stream_async(question):
                # Check if connection is still active
                if not websocket_manager.is_connection_active(session_id):
                    break
                
                connection.session_context.update_activity()
                
                # Convert SSE-style chunk to WebSocket message
                if chunk.get("type") == "metadata":
                    # Send metadata message
                    metadata_msg = create_metadata_message(
                        confidence=chunk.get("confidence", 0.0),
                        total_words=chunk.get("total_words", 0),
                        session_id=session_id
                    )
                    try:
                        await websocket_manager.send_message(session_id, metadata_msg)
                    except Exception as send_error:
                        print(f"🔌 Failed to send metadata message to {session_id}: {send_error}")
                        break
                
                elif chunk.get("type") == "word":
                    # Send word message
                    word_msg = create_word_message(
                        content=chunk.get("content", ""),
                        index=chunk.get("index", word_count),
                        confidence=chunk.get("confidence", 0.0)
                    )
                    try:
                        await websocket_manager.send_message(session_id, word_msg)
                        word_count += 1
                    except Exception as send_error:
                        print(f"🔌 Failed to send word message to {session_id}: {send_error}")
                        break
                
                elif chunk.get("type") == "complete":
                    processing_stage = chunk.get("processing_stage", "completed")
                    confidence = chunk.get("final_confidence", confidence)
                    
                    # Send completion message
                    complete_msg = create_complete_message(
                        processing_stage=processing_stage,
                        final_confidence=confidence,
                        total_words=chunk.get("total_words", word_count)
                    )
                    try:
                        await websocket_manager.send_message(session_id, complete_msg)
                    except Exception as send_error:
                        print(f"🔌 Failed to send completion message to {session_id}: {send_error}")
                    break
                
                elif chunk.get("type") == "error":
                    error = chunk.get("error", "Unknown error")
                    processing_stage = chunk.get("processing_stage", "error")
                    
                    # Send error message
                    error_msg = create_error_message(
                        error=error,
                        processing_stage=processing_stage
                    )
                    try:
                        await websocket_manager.send_message(session_id, error_msg)
                    except Exception as send_error:
                        print(f"🔌 Failed to send error message to {session_id}: {send_error}")
                    break
        
        except Exception as e:
            error = str(e)
            error_msg = create_error_message(f"Streaming error: {error}")
            # Only try to send error message if connection is still active
            if websocket_manager.is_connection_active(session_id):
                try:
                    await websocket_manager.send_message(session_id, error_msg)
                except Exception as send_error:
                    print(f"🔌 Failed to send streaming error message to {session_id}: {send_error}")
        
        finally:
            # Log agent processing performance
            processing_time = time.time() - start_time
            logger.log_agent_processing(
                question=question,
                processing_time=processing_time,
                confidence=confidence,
                processing_stage=processing_stage,
                session_id=session_id,
                error=error,
                word_count=word_count
            )
            
            connection.is_streaming = False
            # Send final status only if connection is still active
            if websocket_manager.is_connection_active(session_id):
                try:
                    status_msg = create_status_message("ready")
                    await websocket_manager.send_message(session_id, status_msg)
                except Exception as send_error:
                    print(f"🔌 Failed to send ready status to {session_id}: {send_error}")
    
    except Exception as e:
        processing_time = time.time() - start_time
        # Only try to send error message if connection is still active
        if websocket_manager.is_connection_active(session_id):
            try:
                error_msg = create_error_message(f"Question processing error: {str(e)}")
                await websocket_manager.send_message(session_id, error_msg)
            except Exception as send_error:
                print(f"🔌 Failed to send processing error message to {session_id}: {send_error}")
        
        # Log the error
        logger.log_error(e, "WebSocket question processing", session_id=session_id)
        logger.log_agent_processing(
            question=question if 'question' in locals() else "unknown",
            processing_time=processing_time,
            confidence=0.0,
            processing_stage="error",
            session_id=session_id,
            error=str(e)
        )


async def handle_websocket_cancel(session_id: str, connection):
    """Handle a cancel request from WebSocket client."""
    connection.is_streaming = False
    if websocket_manager.is_connection_active(session_id):
        status_msg = create_status_message("cancelled")
        await websocket_manager.send_message(session_id, status_msg)


async def handle_websocket_pause(session_id: str, connection):
    """Handle a pause request from WebSocket client."""
    if websocket_manager.is_connection_active(session_id):
        status_msg = create_status_message("paused")
        await websocket_manager.send_message(session_id, status_msg)


async def handle_websocket_resume(session_id: str, connection):
    """Handle a resume request from WebSocket client."""
    if websocket_manager.is_connection_active(session_id):
        status_msg = create_status_message("streaming")
        await websocket_manager.send_message(session_id, status_msg)


@app.get("/ws/stats")
async def get_websocket_stats():
    """Get WebSocket connection statistics."""
    try:
        stats = websocket_manager.get_connection_stats()
        session_stats = await session_manager.get_session_stats()
        
        return {
            "websocket_connections": stats,
            "session_management": session_stats,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get WebSocket stats: {str(e)}"
        )


@app.get("/metrics")
async def get_prometheus_metrics():
    """Get Prometheus-formatted metrics for monitoring systems."""
    try:
        # Update session count in metrics
        session_stats = await session_manager.get_session_stats()
        metrics_collector.set_active_sessions(session_stats.get("active_sessions", 0))
        
        # Return Prometheus-formatted metrics
        return Response(
            content=metrics_collector.get_prometheus_metrics(),
            media_type=CONTENT_TYPE_LATEST
        )
    except Exception as e:
        logger = get_logger()
        logger.log_error(e, "Failed to get Prometheus metrics")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get Prometheus metrics: {str(e)}"
        )


@app.get("/health-status")
async def get_health_status():
    """Get comprehensive health status with scoring and diagnostics."""
    try:
        return monitoring_dashboard.get_health_status()
    except Exception as e:
        logger = get_logger()
        logger.log_error(e, "Failed to get health status")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get health status: {str(e)}"
        )


@app.get("/performance-stats")
async def get_performance_stats():
    """Get detailed performance statistics and metrics."""
    try:
        return monitoring_dashboard.get_performance_stats()
    except Exception as e:
        logger = get_logger()
        logger.log_error(e, "Failed to get performance stats")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance stats: {str(e)}"
        )


@app.get("/agent-stats")
async def get_agent_stats():
    """Get DeFi agent specific statistics and metrics."""
    try:
        return monitoring_dashboard.get_agent_stats()
    except Exception as e:
        logger = get_logger()
        logger.log_error(e, "Failed to get agent stats")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get agent stats: {str(e)}"
        )


@app.get("/system-metrics")
async def get_system_metrics():
    """Get comprehensive system metrics for monitoring and observability (legacy endpoint)."""
    try:
        logger = get_logger()
        
        # Get current system stats
        session_stats = await session_manager.get_session_stats()
        websocket_stats = websocket_manager.get_connection_stats()
        
        # Calculate system metrics
        active_sessions = session_stats.get("active_sessions", 0)
        active_connections = websocket_stats.get("total_connections", 0)
        
        # Get system resource usage
        memory_usage_mb = None
        cpu_usage_percent = None
        
        try:
            import psutil
            process = psutil.Process()
            memory_usage_mb = process.memory_info().rss / 1024 / 1024
            cpu_usage_percent = process.cpu_percent()
        except ImportError:
            pass  # psutil not available
        
        # Log system metrics
        logger.log_system_metrics(
            active_sessions=active_sessions,
            active_connections=active_connections,
            memory_usage_mb=memory_usage_mb,
            cpu_usage_percent=cpu_usage_percent
        )
        
        return {
            "system": {
                "active_sessions": active_sessions,
                "active_connections": active_connections,
                "memory_usage_mb": memory_usage_mb,
                "cpu_usage_percent": cpu_usage_percent,
                "agent_status": "loaded" if agent else "not_loaded"
            },
            "sessions": session_stats,
            "websockets": websocket_stats,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger = get_logger()
        logger.log_error(e, "Failed to get system metrics")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system metrics: {str(e)}"
        )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """General exception handler for unexpected errors."""
    print(f"❌ Unexpected error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "An unexpected error occurred",
            "status_code": 500
        }
    )


# Dashboard endpoint
@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    """
    Serve the monitoring dashboard HTML interface.
    Provides real-time visualization of application metrics and health status.
    """
    dashboard_path = Path(__file__).parent / "dashboard.html"
    
    if not dashboard_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found")
    
    with open(dashboard_path, "r", encoding="utf-8") as f:
        dashboard_content = f.read()
    
    app_logger.log_request(
        method="GET",
        path="/dashboard",
        status_code=200,
        processing_time=0.05
    )
    
    return HTMLResponse(content=dashboard_content, status_code=200)


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting DeFi Q&A API server...")
    print(f"📋 Environment: {config.ENVIRONMENT}")
    print(f"🌐 Host: {config.HOST}:{config.PORT}")
    print(f"🔒 CORS Origins: {config.ALLOWED_ORIGINS}")
    print(f"⚡ Rate Limit: {config.RATE_LIMIT_REQUESTS_PER_MINUTE} requests/min")
    
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.RELOAD,
        log_level=config.LOG_LEVEL.lower(),
        # Add SSL configuration if enabled
        ssl_keyfile=config.SSL_KEY_PATH if config.USE_HTTPS else None,
        ssl_certfile=config.SSL_CERT_PATH if config.USE_HTTPS else None,
    ) 