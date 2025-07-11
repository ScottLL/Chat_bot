"""
Logging Configuration for DeFi Q&A Application

This module provides comprehensive logging setup using loguru for structured logging,
performance monitoring, and error tracking. Designed for production environments with
proper log rotation, filtering, and JSON formatting for easy integration with monitoring systems.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional
from loguru import logger


class LogConfig:
    """Centralized logging configuration for the DeFi Q&A application."""
    
    def __init__(
        self,
        log_level: str = "INFO",
        log_dir: str = "logs",
        max_file_size: str = "100 MB",
        retention: str = "30 days",
        compression: str = "gz",
        enable_json_logs: bool = True,
        enable_console_logs: bool = True
    ):
        self.log_level = log_level.upper()
        self.log_dir = Path(log_dir)
        self.max_file_size = max_file_size
        self.retention = retention
        self.compression = compression
        self.enable_json_logs = enable_json_logs
        self.enable_console_logs = enable_console_logs
        
        # Ensure log directory exists
        self.log_dir.mkdir(exist_ok=True)
        
        # Configure loguru
        self._setup_loggers()
    
    def _setup_loggers(self):
        """Configure loguru loggers with different outputs and formats."""
        
        # Remove default logger
        logger.remove()
        
        # Console logging (development and debugging)
        if self.enable_console_logs:
            logger.add(
                sys.stdout,
                level=self.log_level,
                format=self._get_console_format(),
                colorize=True,
                filter=self._console_filter
            )
        
        # General application logs (JSON format for production)
        if self.enable_json_logs:
            logger.add(
                self.log_dir / "app.log",
                level=self.log_level,
                format=self._get_json_format(),
                rotation=self.max_file_size,
                retention=self.retention,
                compression=self.compression,
                serialize=True,  # JSON serialization
                enqueue=True     # Thread-safe logging
            )
        
        # Error-only logs (for monitoring and alerting)
        logger.add(
            self.log_dir / "errors.log",
            level="ERROR",
            format=self._get_json_format(),
            rotation=self.max_file_size,
            retention=self.retention,
            compression=self.compression,
            serialize=True,
            enqueue=True
        )
        
        # Performance logs (for metrics and monitoring)
        logger.add(
            self.log_dir / "performance.log",
            level="INFO",
            format=self._get_json_format(),
            rotation=self.max_file_size,
            retention=self.retention,
            compression=self.compression,
            serialize=True,
            enqueue=True,
            filter=lambda record: "performance" in record.get("extra", {}).get("category", "")
        )
        
        # WebSocket connection logs
        logger.add(
            self.log_dir / "websocket.log",
            level="DEBUG",
            format=self._get_json_format(),
            rotation=self.max_file_size,
            retention=self.retention,
            compression=self.compression,
            serialize=True,
            enqueue=True,
            filter=lambda record: "websocket" in record.get("extra", {}).get("category", "")
        )
    
    def _get_console_format(self) -> str:
        """Get console format for development."""
        return (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    def _get_json_format(self) -> str:
        """Get JSON format for production logs."""
        return "{message}"
    
    def _console_filter(self, record):
        """Filter console logs to avoid noise in development."""
        # Skip some noisy logs in development
        if record["level"].no < 20:  # Skip DEBUG logs in console
            return False
        return True


class ApplicationLogger:
    """Application-specific logger with structured logging methods."""
    
    def __init__(self):
        self.logger = logger
    
    def log_request(
        self,
        method: str,
        path: str,
        status_code: int,
        processing_time: float,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs
    ):
        """Log HTTP request with performance metrics."""
        self.logger.info(
            "HTTP Request",
            extra={
                "category": "performance",
                "event_type": "http_request",
                "method": method,
                "path": path,
                "status_code": status_code,
                "processing_time_ms": round(processing_time * 1000, 2),
                "user_id": user_id,
                "session_id": session_id,
                **kwargs
            }
        )
    
    def log_websocket_event(
        self,
        event_type: str,
        session_id: str,
        connection_count: int = None,
        message_type: str = None,
        **kwargs
    ):
        """Log WebSocket events."""
        self.logger.info(
            f"WebSocket {event_type}",
            extra={
                "category": "websocket",
                "event_type": event_type,
                "session_id": session_id,
                "connection_count": connection_count,
                "message_type": message_type,
                **kwargs
            }
        )
    
    def log_agent_processing(
        self,
        question: str,
        processing_time: float,
        confidence: float,
        processing_stage: str,
        session_id: str,
        error: Optional[str] = None,
        **kwargs
    ):
        """Log DeFi agent processing metrics."""
        level = "ERROR" if error else "INFO"
        self.logger.log(
            level,
            f"Agent Processing {'Failed' if error else 'Completed'}",
            extra={
                "category": "performance",
                "event_type": "agent_processing",
                "question_length": len(question),
                "processing_time_ms": round(processing_time * 1000, 2),
                "confidence": round(confidence, 3) if confidence else None,
                "processing_stage": processing_stage,
                "session_id": session_id,
                "error": error,
                **kwargs
            }
        )
    
    def log_system_metrics(
        self,
        active_sessions: int,
        active_connections: int,
        memory_usage_mb: float = None,
        cpu_usage_percent: float = None,
        **kwargs
    ):
        """Log system performance metrics."""
        self.logger.info(
            "System Metrics",
            extra={
                "category": "performance",
                "event_type": "system_metrics",
                "active_sessions": active_sessions,
                "active_connections": active_connections,
                "memory_usage_mb": round(memory_usage_mb, 2) if memory_usage_mb else None,
                "cpu_usage_percent": round(cpu_usage_percent, 2) if cpu_usage_percent else None,
                **kwargs
            }
        )
    
    def log_error(
        self,
        error: Exception,
        context: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs
    ):
        """Log application errors with context."""
        self.logger.error(
            f"Application Error: {context}",
            extra={
                "category": "error",
                "event_type": "application_error",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context,
                "session_id": session_id,
                "user_id": user_id,
                **kwargs
            }
        )
    
    def log_startup(self, component: str, success: bool = True, **kwargs):
        """Log application startup events."""
        level = "INFO" if success else "ERROR"
        self.logger.log(
            level,
            f"Startup: {component} {'✅ Success' if success else '❌ Failed'}",
            extra={
                "category": "system",
                "event_type": "startup",
                "component": component,
                "success": success,
                **kwargs
            }
        )
    
    def log_shutdown(self, component: str, **kwargs):
        """Log application shutdown events."""
        self.logger.info(
            f"Shutdown: {component}",
            extra={
                "category": "system",
                "event_type": "shutdown",
                "component": component,
                **kwargs
            }
        )


def setup_logging(
    log_level: str = None,
    log_dir: str = None,
    enable_json_logs: bool = None,
    enable_console_logs: bool = None
) -> ApplicationLogger:
    """
    Setup logging configuration for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files
        enable_json_logs: Whether to enable JSON formatted file logs
        enable_console_logs: Whether to enable console logging
    
    Returns:
        ApplicationLogger instance for structured logging
    """
    # Get configuration from environment variables with defaults
    config = LogConfig(
        log_level=log_level or os.getenv("LOG_LEVEL", "INFO"),
        log_dir=log_dir or os.getenv("LOG_DIR", "logs"),
        enable_json_logs=enable_json_logs if enable_json_logs is not None else 
                         os.getenv("ENABLE_JSON_LOGS", "true").lower() == "true",
        enable_console_logs=enable_console_logs if enable_console_logs is not None else 
                           os.getenv("ENABLE_CONSOLE_LOGS", "true").lower() == "true"
    )
    
    return ApplicationLogger()


# Global logger instance
app_logger = setup_logging()


def get_logger() -> ApplicationLogger:
    """Get the global application logger instance."""
    return app_logger


# Utility functions for common logging patterns
def log_performance(func_name: str, execution_time: float, **kwargs):
    """Helper function to log performance metrics."""
    app_logger.logger.info(
        f"Performance: {func_name}",
        extra={
            "category": "performance",
            "event_type": "function_execution",
            "function": func_name,
            "execution_time_ms": round(execution_time * 1000, 2),
            **kwargs
        }
    )


def log_api_call(api_name: str, success: bool, response_time: float, **kwargs):
    """Helper function to log external API calls."""
    level = "INFO" if success else "WARNING"
    app_logger.logger.log(
        level,
        f"API Call: {api_name} {'Success' if success else 'Failed'}",
        extra={
            "category": "external_api",
            "event_type": "api_call",
            "api_name": api_name,
            "success": success,
            "response_time_ms": round(response_time * 1000, 2),
            **kwargs
        }
    ) 