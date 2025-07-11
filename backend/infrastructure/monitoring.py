"""
Monitoring and Metrics Module for DeFi Q&A Application

This module provides comprehensive monitoring capabilities including:
- Prometheus-compatible metrics
- Real-time application statistics
- Performance tracking and alerting
- System health monitoring
- Custom metrics for DeFi Q&A specific operations

Integrates with the existing logging framework for full observability.
"""

import time
import psutil
import asyncio
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
from datetime import datetime, timedelta
from prometheus_client import (
    Counter, Histogram, Gauge, Info, 
    generate_latest, CONTENT_TYPE_LATEST
)
from prometheus_client.core import CollectorRegistry
from .logging_config import get_logger


class MetricsCollector:
    """Centralized metrics collection for Prometheus and custom monitoring."""
    
    def __init__(self):
        # Create a custom registry for our metrics
        self.registry = CollectorRegistry()
        
        # HTTP Request Metrics
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.http_request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # WebSocket Metrics
        self.websocket_connections_total = Counter(
            'websocket_connections_total',
            'Total WebSocket connections',
            ['status'],  # connected, disconnected, error
            registry=self.registry
        )
        
        self.websocket_active_connections = Gauge(
            'websocket_active_connections',
            'Currently active WebSocket connections',
            registry=self.registry
        )
        
        self.websocket_messages_total = Counter(
            'websocket_messages_total',
            'Total WebSocket messages',
            ['direction', 'type'],  # sent/received, question/response/error
            registry=self.registry
        )
        
        # DeFi Agent Metrics
        self.agent_queries_total = Counter(
            'agent_queries_total',
            'Total queries processed by the DeFi agent',
            ['status'],  # completed, error, timeout
            registry=self.registry
        )
        
        self.agent_processing_duration = Histogram(
            'agent_processing_duration_seconds',
            'Time taken to process DeFi queries',
            ['query_type'],
            registry=self.registry
        )
        
        self.agent_confidence_score = Histogram(
            'agent_confidence_score',
            'Confidence scores for agent responses',
            buckets=[0.1, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0],
            registry=self.registry
        )
        
        # System Metrics
        self.system_memory_usage = Gauge(
            'system_memory_usage_bytes',
            'Current memory usage in bytes',
            registry=self.registry
        )
        
        self.system_cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'Current CPU usage percentage',
            registry=self.registry
        )
        
        self.active_sessions = Gauge(
            'active_sessions_total',
            'Number of active user sessions',
            registry=self.registry
        )
        
        # Application Info
        self.app_info = Info(
            'app_info',
            'Application information',
            registry=self.registry
        )
        
        # Error Metrics
        self.errors_total = Counter(
            'errors_total',
            'Total application errors',
            ['error_type', 'component'],
            registry=self.registry
        )
        
        # Performance tracking for custom monitoring
        self.recent_requests = deque(maxlen=1000)  # Keep last 1000 requests
        self.response_times = defaultdict(list)    # Response times by endpoint
        self.error_counts = defaultdict(int)       # Error counts by type
        self.start_time = time.time()
        
        # Initialize app info
        self.app_info.info({
            'version': '1.0.0',
            'name': 'DeFi Q&A API',
            'description': 'Semantic search API for DeFi questions'
        })
        
        self.logger = get_logger()
    
    def record_http_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics."""
        # Prometheus metrics
        self.http_requests_total.labels(
            method=method, 
            endpoint=endpoint, 
            status=str(status_code)
        ).inc()
        
        self.http_request_duration.labels(
            method=method, 
            endpoint=endpoint
        ).observe(duration)
        
        # Custom tracking
        request_data = {
            'timestamp': time.time(),
            'method': method,
            'endpoint': endpoint,
            'status_code': status_code,
            'duration': duration
        }
        self.recent_requests.append(request_data)
        self.response_times[endpoint].append(duration)
        
        # Keep only recent response times (last 100 per endpoint)
        if len(self.response_times[endpoint]) > 100:
            self.response_times[endpoint] = self.response_times[endpoint][-100:]
    
    def record_websocket_connection(self, status: str, active_count: int):
        """Record WebSocket connection metrics."""
        self.websocket_connections_total.labels(status=status).inc()
        self.websocket_active_connections.set(active_count)
    
    def record_websocket_message(self, direction: str, message_type: str):
        """Record WebSocket message metrics."""
        self.websocket_messages_total.labels(
            direction=direction,
            type=message_type
        ).inc()
    
    def record_agent_query(self, status: str, duration: float, confidence: float = None, query_type: str = "general"):
        """Record DeFi agent query metrics."""
        self.agent_queries_total.labels(status=status).inc()
        self.agent_processing_duration.labels(query_type=query_type).observe(duration)
        
        if confidence is not None:
            self.agent_confidence_score.observe(confidence)
    
    def record_error(self, error_type: str, component: str):
        """Record error metrics."""
        self.errors_total.labels(error_type=error_type, component=component).inc()
        self.error_counts[f"{component}:{error_type}"] += 1
    
    def update_system_metrics(self):
        """Update system resource metrics."""
        try:
            # Memory usage
            memory_info = psutil.virtual_memory()
            self.system_memory_usage.set(memory_info.used)
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            self.system_cpu_usage.set(cpu_percent)
            
        except Exception as e:
            self.logger.log_error(e, "Failed to update system metrics")
    
    def set_active_sessions(self, count: int):
        """Update active sessions count."""
        self.active_sessions.set(count)
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus-formatted metrics."""
        return generate_latest(self.registry)


class MonitoringDashboard:
    """Real-time monitoring dashboard for application statistics."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.logger = get_logger()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get application health status."""
        try:
            # Calculate uptime
            uptime_seconds = time.time() - self.metrics.start_time
            uptime_formatted = str(timedelta(seconds=int(uptime_seconds)))
            
            # System health checks
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Application health indicators
            recent_errors = sum(self.metrics.error_counts.values())
            active_connections = self.metrics.websocket_active_connections._value._value
            
            # Determine overall health
            health_score = 100
            health_issues = []
            
            # Check memory usage (warn if > 80%, critical if > 95%)
            if memory_info.percent > 95:
                health_score -= 30
                health_issues.append(f"Critical memory usage: {memory_info.percent:.1f}%")
            elif memory_info.percent > 80:
                health_score -= 15
                health_issues.append(f"High memory usage: {memory_info.percent:.1f}%")
            
            # Check CPU usage (warn if > 80%, critical if > 95%)
            if cpu_percent > 95:
                health_score -= 25
                health_issues.append(f"Critical CPU usage: {cpu_percent:.1f}%")
            elif cpu_percent > 80:
                health_score -= 10
                health_issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            # Check recent errors (warn if > 10 in last hour)
            if recent_errors > 10:
                health_score -= 20
                health_issues.append(f"High error rate: {recent_errors} recent errors")
            
            # Determine status
            if health_score >= 90:
                status = "healthy"
            elif health_score >= 70:
                status = "warning"
            else:
                status = "critical"
            
            return {
                "status": status,
                "health_score": health_score,
                "uptime": uptime_formatted,
                "uptime_seconds": uptime_seconds,
                "system": {
                    "memory_usage_percent": memory_info.percent,
                    "memory_used_mb": memory_info.used / 1024 / 1024,
                    "memory_total_mb": memory_info.total / 1024 / 1024,
                    "cpu_usage_percent": cpu_percent
                },
                "application": {
                    "active_websocket_connections": active_connections,
                    "recent_errors": recent_errors,
                    "total_requests": len(self.metrics.recent_requests)
                },
                "issues": health_issues,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.log_error(e, "Failed to get health status")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        try:
            # Calculate request statistics
            now = time.time()
            last_hour = now - 3600
            last_minute = now - 60
            
            recent_hour = [r for r in self.metrics.recent_requests if r['timestamp'] > last_hour]
            recent_minute = [r for r in self.metrics.recent_requests if r['timestamp'] > last_minute]
            
            # Response time statistics
            endpoint_stats = {}
            for endpoint, times in self.metrics.response_times.items():
                if times:
                    endpoint_stats[endpoint] = {
                        "avg_response_time": sum(times) / len(times),
                        "min_response_time": min(times),
                        "max_response_time": max(times),
                        "request_count": len(times)
                    }
            
            # Error rate calculation
            total_requests = len(self.metrics.recent_requests)
            error_requests = len([r for r in self.metrics.recent_requests if r['status_code'] >= 400])
            error_rate = (error_requests / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "requests": {
                    "total": total_requests,
                    "last_hour": len(recent_hour),
                    "last_minute": len(recent_minute),
                    "requests_per_minute": len(recent_minute),
                    "requests_per_hour": len(recent_hour)
                },
                "performance": {
                    "error_rate_percent": error_rate,
                    "endpoints": endpoint_stats
                },
                "errors": dict(self.metrics.error_counts),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.log_error(e, "Failed to get performance stats")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get DeFi agent specific statistics."""
        try:
            # This would be populated by actual agent metrics
            # For now, return basic structure
            return {
                "queries": {
                    "total_processed": 0,  # Would be tracked by agent
                    "avg_confidence": 0.0,
                    "avg_processing_time": 0.0
                },
                "categories": {
                    "defi_general": 0,
                    "yield_farming": 0,
                    "lending": 0,
                    "dex": 0,
                    "other": 0
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.log_error(e, "Failed to get agent stats")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


class PerformanceMonitor:
    """Background performance monitoring and alerting."""
    
    def __init__(self, metrics_collector: MetricsCollector, monitoring_dashboard: MonitoringDashboard):
        self.metrics = metrics_collector
        self.dashboard = monitoring_dashboard
        self.logger = get_logger()
        self.monitoring_task = None
        self.alert_thresholds = {
            "memory_usage_critical": 95,
            "memory_usage_warning": 80,
            "cpu_usage_critical": 95,
            "cpu_usage_warning": 80,
            "error_rate_critical": 10,  # errors per minute
            "response_time_warning": 5.0  # seconds
        }
    
    async def start_monitoring(self):
        """Start background monitoring task."""
        if self.monitoring_task is None:
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.logger.log_startup("Performance Monitor", success=True)
    
    async def stop_monitoring(self):
        """Stop background monitoring task."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
            self.logger.log_shutdown("Performance Monitor")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while True:
            try:
                # Update system metrics
                self.metrics.update_system_metrics()
                
                # Check for alerts
                await self._check_alerts()
                
                # Log system metrics periodically
                self.logger.log_system_metrics(
                    active_sessions=self.metrics.active_sessions._value._value,
                    active_connections=self.metrics.websocket_active_connections._value._value,
                    memory_usage_mb=psutil.virtual_memory().used / 1024 / 1024,
                    cpu_usage_percent=psutil.cpu_percent(interval=None)
                )
                
                # Wait 30 seconds before next check
                await asyncio.sleep(30)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.log_error(e, "Error in monitoring loop")
                await asyncio.sleep(30)  # Continue monitoring even if there are errors
    
    async def _check_alerts(self):
        """Check for alert conditions."""
        try:
            health_status = self.dashboard.get_health_status()
            
            # Check for critical conditions
            if health_status["status"] == "critical":
                self.logger.error(f"🚨 CRITICAL ALERT: {', '.join(health_status['issues'])}")
            elif health_status["status"] == "warning":
                self.logger.warning(f"⚠️ WARNING: {', '.join(health_status['issues'])}")
                
        except Exception as e:
            self.logger.log_error(e, "Failed to check alerts")


# Global monitoring instances
metrics_collector = MetricsCollector()
monitoring_dashboard = MonitoringDashboard(metrics_collector)
performance_monitor = PerformanceMonitor(metrics_collector, monitoring_dashboard) 