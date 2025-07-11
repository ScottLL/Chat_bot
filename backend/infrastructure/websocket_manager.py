"""
WebSocket Connection Manager for DeFi Q&A Streaming

This module provides WebSocket connection management with session integration,
enabling real-time bidirectional communication for streaming Q&A responses.
"""

import json
import time
import asyncio
import logging
from typing import Dict, Optional, Set, Any, TYPE_CHECKING
from fastapi import WebSocket, WebSocketDisconnect
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

# Import services
from services.session_manager import SessionContext

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from agents.async_defi_qa_agent import AsyncDeFiQAAgent


@dataclass
class WebSocketConnection:
    """Represents an active WebSocket connection with session context."""
    
    websocket: WebSocket
    session_id: str
    session_context: SessionContext
    agent: "AsyncDeFiQAAgent"
    connected_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    is_streaming: bool = False
    _disconnecting: bool = field(default=False, init=False)  # Track if disconnect is in progress
    
    def update_heartbeat(self):
        """Update the last heartbeat timestamp."""
        self.last_heartbeat = time.time()
    
    def mark_disconnecting(self):
        """Mark connection as disconnecting to prevent new messages."""
        self._disconnecting = True
    
    def is_disconnecting(self) -> bool:
        """Check if connection is in the process of disconnecting."""
        return self._disconnecting


class WebSocketConnectionManager:
    """
    Manages WebSocket connections with session integration.
    
    Features:
    - Session-scoped WebSocket connections
    - Automatic connection cleanup
    - Heartbeat monitoring
    - Integration with existing session management
    - Error handling and recovery
    - Graceful handling of abrupt disconnections
    """
    
    def __init__(self, heartbeat_interval: int = 30, cleanup_interval: int = 60):
        self.active_connections: Dict[str, WebSocketConnection] = {}
        self.connection_by_websocket: Dict[WebSocket, str] = {}
        self.heartbeat_interval = heartbeat_interval
        self.cleanup_interval = cleanup_interval
        self._cleanup_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self.logger = logging.getLogger(__name__)
    
    async def start_background_tasks(self):
        """Start background tasks for connection management."""
        if not self._cleanup_task or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        if not self._heartbeat_task or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
    
    async def stop_background_tasks(self):
        """Stop background tasks."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
    
    async def connect(self, websocket: WebSocket, session_context: SessionContext, agent: "AsyncDeFiQAAgent") -> str:
        """
        Accept a WebSocket connection and associate it with a session.
        
        Args:
            websocket: The WebSocket connection
            session_context: Session context for the connection
            agent: The async agent instance for this session
            
        Returns:
            str: The session ID for the connection
        """
        await websocket.accept()
        
        connection = WebSocketConnection(
            websocket=websocket,
            session_id=session_context.session_id,
            session_context=session_context,
            agent=agent
        )
        
        # Store connection mappings
        self.active_connections[session_context.session_id] = connection
        self.connection_by_websocket[websocket] = session_context.session_id
        
        self.logger.info(f"WebSocket connected for session {session_context.session_id}")
        
        # Start background tasks if not already running
        await self.start_background_tasks()
        
        return session_context.session_id
    
    def _is_websocket_disconnected_error(self, error: Exception) -> bool:
        """
        Check if an error indicates a WebSocket disconnection.
        
        This helps distinguish between expected disconnection errors and real errors.
        """
        error_str = str(error).lower()
        
        # Common WebSocket disconnection error patterns
        disconnection_patterns = [
            "no close frame received or sent",
            "connection closed",
            "websocket connection is closed",
            "connection is closed",
            "close frame received",
            "connection reset",
            "broken pipe",
            "connection aborted"
        ]
        
        return any(pattern in error_str for pattern in disconnection_patterns)
    
    async def disconnect(self, websocket: WebSocket):
        """
        Disconnect a WebSocket connection and clean up resources.
        
        Args:
            websocket: The WebSocket connection to disconnect
        """
        session_id = self.connection_by_websocket.get(websocket)
        if not session_id:
            return
        
        connection = self.active_connections.get(session_id)
        if connection:
            # Mark as disconnecting to prevent new messages
            connection.mark_disconnecting()
        
        # Clean up mappings
        self.active_connections.pop(session_id, None)
        self.connection_by_websocket.pop(websocket, None)
        
        self.logger.info(f"WebSocket disconnected for session {session_id}")
        
        # Close the WebSocket if not already closed
        try:
            # Check multiple possible disconnected states
            if hasattr(websocket, 'client_state') and websocket.client_state.name not in ["DISCONNECTED", "CLOSED"]:
                await websocket.close(code=1000, reason="Server cleanup")
        except Exception as e:
            # Only log as debug if it's a disconnection-related error
            if self._is_websocket_disconnected_error(e):
                self.logger.debug(f"Expected disconnection error during cleanup for session {session_id}: {e}")
            else:
                self.logger.warning(f"Unexpected error closing WebSocket for session {session_id}: {e}")
    
    async def send_message(self, session_id: str, message: Dict[str, Any]) -> bool:
        """
        Send a message to a specific session's WebSocket connection.
        
        Args:
            session_id: The session ID to send the message to
            message: The message data to send
            
        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        connection = self.active_connections.get(session_id)
        if not connection:
            return False
        
        # Check if connection is being disconnected
        if connection.is_disconnecting():
            return False
        
        # Check if WebSocket is still connected before sending
        try:
            # Check WebSocket state if available
            if hasattr(connection.websocket, 'client_state'):
                state = connection.websocket.client_state.name
                if state in ["DISCONNECTED", "CLOSED", "CLOSING"]:
                    # Connection is already closed, clean up silently
                    await self.disconnect(connection.websocket)
                    return False
            
            await connection.websocket.send_text(json.dumps(message))
            connection.update_heartbeat()
            return True
            
        except Exception as e:
            # Handle disconnection errors gracefully
            if self._is_websocket_disconnected_error(e):
                self.logger.debug(f"Client disconnected during message send to session {session_id}: {e}")
            else:
                self.logger.error(f"Error sending message to session {session_id}: {e}")
            
            # Mark for cleanup regardless of error type
            await self.disconnect(connection.websocket)
            return False
    
    async def send_heartbeat(self, session_id: str) -> bool:
        """
        Send a heartbeat message to a specific session.
        
        Args:
            session_id: The session ID to send heartbeat to
            
        Returns:
            bool: True if heartbeat was sent successfully, False otherwise
        """
        heartbeat_message = {
            "type": "heartbeat",
            "timestamp": time.time()
        }
        return await self.send_message(session_id, heartbeat_message)
    
    async def broadcast_to_all(self, message: Dict[str, Any]):
        """
        Broadcast a message to all active connections.
        
        Args:
            message: The message to broadcast
        """
        if not self.active_connections:
            return
        
        # Use asyncio.gather to send to all connections concurrently
        tasks = [
            self.send_message(session_id, message)
            for session_id in list(self.active_connections.keys())
        ]
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_connection(self, session_id: str) -> Optional[WebSocketConnection]:
        """Get connection info for a specific session."""
        return self.active_connections.get(session_id)
    
    def get_connection_by_websocket(self, websocket: WebSocket) -> Optional[WebSocketConnection]:
        """Get connection info by WebSocket instance."""
        session_id = self.connection_by_websocket.get(websocket)
        return self.active_connections.get(session_id) if session_id else None
    
    def is_connection_active(self, session_id: str) -> bool:
        """
        Check if a WebSocket connection is still active and connected.
        
        Args:
            session_id: The session ID to check
            
        Returns:
            bool: True if connection exists and is connected, False otherwise
        """
        connection = self.active_connections.get(session_id)
        if not connection:
            return False
        
        # Check if connection is being disconnected
        if connection.is_disconnecting():
            return False
        
        try:
            # Check if WebSocket is still connected (not in any disconnected state)
            if hasattr(connection.websocket, 'client_state'):
                state = connection.websocket.client_state.name
                return state not in ["DISCONNECTED", "CLOSED", "CLOSING"]
            # If no client_state attribute, assume connection is active
            return True
        except Exception:
            return False
    
    def get_connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self.active_connections)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get detailed connection statistics."""
        now = time.time()
        connections = list(self.active_connections.values())
        
        if not connections:
            return {
                "total_connections": 0,
                "streaming_connections": 0,
                "average_connection_duration": 0,
                "oldest_connection_age": 0
            }
        
        streaming_count = sum(1 for conn in connections if conn.is_streaming)
        durations = [now - conn.connected_at for conn in connections]
        
        return {
            "total_connections": len(connections),
            "streaming_connections": streaming_count,
            "average_connection_duration": sum(durations) / len(durations),
            "oldest_connection_age": max(durations)
        }
    
    async def _cleanup_loop(self):
        """Background task to clean up stale connections."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_stale_connections()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
    
    async def _heartbeat_loop(self):
        """Background task to send heartbeats to active connections."""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                await self._send_heartbeats()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
    
    async def _cleanup_stale_connections(self):
        """Clean up connections that haven't responded to heartbeats."""
        now = time.time()
        stale_threshold = self.heartbeat_interval * 3  # 3 missed heartbeats for more tolerance
        
        stale_sessions = []
        
        # Take a snapshot to avoid modification during iteration
        connections_snapshot = dict(self.active_connections)
        
        for session_id, connection in connections_snapshot.items():
            # Skip if already disconnecting
            if connection.is_disconnecting():
                continue
                
            # Check if connection is stale
            if now - connection.last_heartbeat > stale_threshold:
                # Double-check if connection is still responsive
                if not self.is_connection_active(session_id):
                    stale_sessions.append(session_id)
        
        # Clean up stale connections
        for session_id in stale_sessions:
            connection = self.active_connections.get(session_id)
            if connection and not connection.is_disconnecting():
                self.logger.info(f"Cleaning up stale connection for session {session_id}")
                await self.disconnect(connection.websocket)
        
        if stale_sessions:
            self.logger.debug(f"Cleaned up {len(stale_sessions)} stale connections")
    
    async def _send_heartbeats(self):
        """Send heartbeat messages to all active connections."""
        if not self.active_connections:
            return
        
        # Get snapshot of current connection IDs to avoid modification during iteration
        session_ids = list(self.active_connections.keys())
        
        # Send heartbeats concurrently to all connections
        tasks = [
            self.send_heartbeat(session_id)
            for session_id in session_ids
            # Only send to connections that are still active and not disconnecting
            if self.is_connection_active(session_id)
        ]
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            failed_count = sum(1 for result in results if result is False or isinstance(result, Exception))
            success_count = len(tasks) - failed_count
            
            if failed_count > 0:
                self.logger.debug(f"Heartbeat status: {success_count} sent, {failed_count} failed (likely disconnected clients)")
            else:
                self.logger.debug(f"Heartbeat sent to {success_count} active connections")


# Global WebSocket manager instance
websocket_manager = WebSocketConnectionManager()


@asynccontextmanager
async def websocket_lifespan():
    """Context manager for WebSocket manager lifecycle."""
    try:
        await websocket_manager.start_background_tasks()
        yield websocket_manager
    finally:
        await websocket_manager.stop_background_tasks()


# WebSocket message types and protocols
class WebSocketMessageType:
    """Constants for WebSocket message types."""
    
    # Client to server
    QUESTION = "question"
    CANCEL = "cancel"
    PAUSE = "pause"
    RESUME = "resume"
    HEARTBEAT_RESPONSE = "heartbeat_response"
    
    # Server to client
    WORD = "word"
    METADATA = "metadata"
    COMPLETE = "complete"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    STATUS = "status"


def create_word_message(content: str, index: int, confidence: float, processing_stage: str = "streaming") -> Dict[str, Any]:
    """Create a word streaming message."""
    return {
        "type": WebSocketMessageType.WORD,
        "content": content,
        "index": index,
        "confidence": confidence,
        "processing_stage": processing_stage,
        "timestamp": time.time()
    }


def create_metadata_message(confidence: float, total_words: int, session_id: str, processing_stage: str = "streaming_response") -> Dict[str, Any]:
    """Create a metadata message."""
    return {
        "type": WebSocketMessageType.METADATA,
        "confidence": confidence,
        "processing_stage": processing_stage,
        "total_words": total_words,
        "session_id": session_id,
        "timestamp": time.time()
    }


def create_complete_message(processing_stage: str, final_confidence: float, total_words: int) -> Dict[str, Any]:
    """Create a completion message."""
    return {
        "type": WebSocketMessageType.COMPLETE,
        "processing_stage": processing_stage,
        "final_confidence": final_confidence,
        "total_words": total_words,
        "timestamp": time.time()
    }


def create_error_message(error: str, processing_stage: str = "error") -> Dict[str, Any]:
    """Create an error message."""
    return {
        "type": WebSocketMessageType.ERROR,
        "error": error,
        "processing_stage": processing_stage,
        "timestamp": time.time()
    }


def create_status_message(status: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a status message."""
    message = {
        "type": WebSocketMessageType.STATUS,
        "status": status,
        "timestamp": time.time()
    }
    if details:
        message["details"] = details
    return message 