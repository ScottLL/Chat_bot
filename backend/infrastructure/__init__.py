"""
Infrastructure components for the DeFi Q&A bot.

This package contains infrastructure-level components including WebSocket management,
monitoring, error handling, and logging configuration.
"""

from .websocket_manager import WebSocketConnectionManager
from .monitoring import (
    MetricsCollector, 
    MonitoringDashboard, 
    PerformanceMonitor,
    metrics_collector, 
    monitoring_dashboard, 
    performance_monitor
)
from .error_handlers import (
    ErrorHandler,
    ErrorCategory,
    ErrorSeverity,
    StructuredError,
    UserGuidance,
    format_error_for_response,
    format_error_for_stream
)
from .logging_config import get_logger, setup_logging

__all__ = [
    "WebSocketConnectionManager",
    "MetricsCollector",
    "MonitoringDashboard", 
    "PerformanceMonitor",
    "metrics_collector",
    "monitoring_dashboard",
    "performance_monitor",
    "ErrorHandler",
    "ErrorCategory",
    "ErrorSeverity",
    "StructuredError",
    "UserGuidance",
    "format_error_for_response",
    "format_error_for_stream",
    "get_logger", 
    "setup_logging"
] 