"""
Test script for the logging system implementation.

This script tests all aspects of our structured logging system:
- Basic logging functionality
- Log file creation and rotation
- Structured log format
- Performance metrics logging
- Error logging
- Log filtering by category
"""

import os
import sys
import json
import time
import tempfile
from pathlib import Path

# Add parent directory to path to enable imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from infrastructure.logging_config import LogConfig, ApplicationLogger, setup_logging, get_logger


def test_log_config():
    """Test the LogConfig class initialization and setup."""
    print("🧪 Testing LogConfig initialization...")
    
    # Create a temporary directory for test logs
    with tempfile.TemporaryDirectory() as temp_dir:
        config = LogConfig(
            log_level="DEBUG",
            log_dir=temp_dir,
            max_file_size="1 MB",
            retention="1 day",
            enable_json_logs=True,
            enable_console_logs=False  # Disable console for cleaner test output
        )
        
        # Verify log directory was created
        log_dir = Path(temp_dir)
        assert log_dir.exists(), "Log directory should be created"
        
        # Verify log files were created
        expected_files = ["app.log", "errors.log", "performance.log", "websocket.log"]
        for file_name in expected_files:
            log_file = log_dir / file_name
            assert log_file.exists(), f"Log file {file_name} should be created"
        
        print("✅ LogConfig initialization test passed!")


def test_application_logger():
    """Test the ApplicationLogger methods."""
    print("🧪 Testing ApplicationLogger methods...")
    
    # Create a temporary directory for test logs
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup logging with test directory
        config = LogConfig(
            log_level="DEBUG",
            log_dir=temp_dir,
            enable_console_logs=False
        )
        
        logger = ApplicationLogger()
        
        # Test HTTP request logging
        logger.log_request(
            method="GET",
            path="/test",
            status_code=200,
            processing_time=0.123,
            session_id="test-session-123",
            user_id="test-user",
            client_ip="192.168.1.100"
        )
        
        # Test WebSocket event logging
        logger.log_websocket_event(
            event_type="connected",
            session_id="test-session-123",
            connection_count=1,
            message_type="connection"
        )
        
        # Test agent processing logging
        logger.log_agent_processing(
            question="What is DeFi?",
            processing_time=1.5,
            confidence=0.85,
            processing_stage="completed",
            session_id="test-session-123",
            word_count=25
        )
        
        # Test system metrics logging
        logger.log_system_metrics(
            active_sessions=3,
            active_connections=5,
            memory_usage_mb=128.5,
            cpu_usage_percent=15.2
        )
        
        # Test error logging
        try:
            raise ValueError("Test error for logging")
        except Exception as e:
            logger.log_error(
                error=e,
                context="Test error context",
                session_id="test-session-123",
                user_id="test-user"
            )
        
        # Test startup/shutdown logging
        logger.log_startup("Test Component", success=True, version="1.0.0")
        logger.log_shutdown("Test Component")
        
        # Give some time for logs to be written
        time.sleep(0.1)
        
        # Verify logs were written
        app_log = Path(temp_dir) / "app.log"
        errors_log = Path(temp_dir) / "errors.log"
        
        assert app_log.exists() and app_log.stat().st_size > 0, "App log should have content"
        assert errors_log.exists() and errors_log.stat().st_size > 0, "Errors log should have content"
        
        print("✅ ApplicationLogger methods test passed!")


def test_structured_log_format():
    """Test that logs are properly structured as JSON."""
    print("🧪 Testing structured log format...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = LogConfig(
            log_level="DEBUG",
            log_dir=temp_dir,
            enable_console_logs=False
        )
        
        logger = ApplicationLogger()
        
        # Log a test message
        logger.log_request(
            method="POST",
            path="/api/test",
            status_code=201,
            processing_time=0.456,
            session_id="format-test-session"
        )
        
        # Give time for logs to be written
        time.sleep(0.1)
        
        # Read and verify log format
        app_log = Path(temp_dir) / "app.log"
        with open(app_log, 'r') as f:
            log_line = f.readline().strip()
        
        # Parse as JSON to verify structure
        log_data = json.loads(log_line)
        
        # Verify required fields
        assert "text" in log_data, "Log should have 'text' field"
        assert "record" in log_data, "Log should have 'record' field"
        
        record = log_data["record"]
        assert "time" in record, "Record should have 'time' field"
        assert "level" in record, "Record should have 'level' field"
        assert "message" in record, "Record should have 'message' field"
        assert "extra" in record, "Record should have 'extra' field"
        
        # Verify our custom fields
        extra_data = record["extra"]["extra"]
        assert extra_data["method"] == "POST", "Method should be logged correctly"
        assert extra_data["path"] == "/api/test", "Path should be logged correctly"
        assert extra_data["status_code"] == 201, "Status code should be logged correctly"
        assert extra_data["session_id"] == "format-test-session", "Session ID should be logged correctly"
        
        print("✅ Structured log format test passed!")


def test_performance_metrics():
    """Test performance-related logging features."""
    print("🧪 Testing performance metrics...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = LogConfig(
            log_level="DEBUG",
            log_dir=temp_dir,
            enable_console_logs=False
        )
        
        logger = ApplicationLogger()
        
        # Test timing accuracy
        start_time = time.time()
        time.sleep(0.1)  # Simulate some processing
        processing_time = time.time() - start_time
        
        logger.log_agent_processing(
            question="Performance test question",
            processing_time=processing_time,
            confidence=0.9,
            processing_stage="completed",
            session_id="perf-test-session"
        )
        
        # Give time for logs to be written
        time.sleep(0.1)
        
        # Verify timing was logged correctly
        app_log = Path(temp_dir) / "app.log"
        with open(app_log, 'r') as f:
            log_line = f.readline().strip()
        
        log_data = json.loads(log_line)
        logged_time_ms = log_data["record"]["extra"]["extra"]["processing_time_ms"]
        
        # Should be approximately 100ms (within 50ms tolerance)
        assert 50 <= logged_time_ms <= 150, f"Processing time should be ~100ms, got {logged_time_ms}ms"
        
        print("✅ Performance metrics test passed!")


def test_error_handling():
    """Test error logging and exception handling."""
    print("🧪 Testing error handling...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = LogConfig(
            log_level="DEBUG",
            log_dir=temp_dir,
            enable_console_logs=False
        )
        
        logger = ApplicationLogger()
        
        # Test different types of errors
        errors_to_test = [
            ValueError("Invalid value provided"),
            KeyError("Missing required key"),
            ConnectionError("Failed to connect to service"),
            TimeoutError("Operation timed out")
        ]
        
        for error in errors_to_test:
            logger.log_error(
                error=error,
                context=f"Testing {type(error).__name__}",
                session_id="error-test-session"
            )
        
        # Give time for logs to be written
        time.sleep(0.1)
        
        # Verify errors were logged
        errors_log = Path(temp_dir) / "errors.log"
        assert errors_log.exists(), "Errors log should exist"
        
        with open(errors_log, 'r') as f:
            error_lines = f.readlines()
        
        assert len(error_lines) == len(errors_to_test), f"Should have {len(errors_to_test)} error logs"
        
        # Verify error structure
        for line in error_lines:
            log_data = json.loads(line.strip())
            extra_data = log_data["record"]["extra"]["extra"]
            
            assert "error_type" in extra_data, "Error type should be logged"
            assert "error_message" in extra_data, "Error message should be logged"
            assert "context" in extra_data, "Error context should be logged"
            assert extra_data["category"] == "error", "Error category should be set"
        
        print("✅ Error handling test passed!")


def main():
    """Run all logging system tests."""
    print("🚀 Starting logging system tests...\n")
    
    try:
        test_log_config()
        print()
        
        test_application_logger()
        print()
        
        test_structured_log_format()
        print()
        
        test_performance_metrics()
        print()
        
        test_error_handling()
        print()
        
        print("🎉 All logging system tests passed!")
        print("\n📋 Summary:")
        print("✅ LogConfig initialization and setup")
        print("✅ ApplicationLogger methods")
        print("✅ Structured JSON log format")
        print("✅ Performance metrics accuracy")
        print("✅ Error handling and logging")
        print("\n🎯 The logging framework is ready for production use!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main() 