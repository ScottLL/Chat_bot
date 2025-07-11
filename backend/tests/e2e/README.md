# End-to-End (E2E) Tests

This directory contains comprehensive end-to-end tests for production readiness, deployment validation, and system-wide functionality testing.

## 🧪 Test Suite Overview

### 1. **Deployment Testing** (`test_deployment.py`)
**Purpose**: Validates deployment configurations and production readiness
- ✅ Configuration file validation (Procfile, docker files, etc.)
- ✅ Environment variable verification  
- ✅ API endpoint health checks
- ✅ Monitoring endpoints validation
- ✅ CORS configuration testing
- ✅ Rate limiting verification
- ✅ Docker build testing

**Usage:**
```bash
# Test local deployment
python test_deployment.py

# Test production deployment
python test_deployment.py --url https://your-domain.com

# Skip Docker tests
python test_deployment.py --skip-docker
```

### 2. **Thread Safety & Concurrency** (`test_thread_safety.py`)
**Purpose**: Validates system behavior under concurrent load
- 🔒 Concurrent request handling (50+ requests)
- 🔗 Session isolation testing
- 📡 Streaming concurrency validation
- ⚡ Rate limiting behavior under load
- 📊 Performance metrics collection

**Usage:**
```bash
# Run comprehensive thread safety tests
python test_thread_safety.py

# Results saved to: thread_safety_test_results.json
```

### 3. **Error Handling** (`test_error_handling.py`)
**Purpose**: Validates error scenarios and user-friendly error responses
- ❌ Validation error handling
- 🔍 No results scenarios
- 📡 Streaming error recovery
- 🛡️ Graceful degradation testing

**Usage:**
```bash
# Test error handling scenarios
python test_error_handling.py
```

### 4. **WebSocket Client** (`test_websocket_client.py`)
**Purpose**: Client-side WebSocket functionality testing
- 🔗 Connection establishment
- 📤 Message sending/receiving
- 📡 Real-time streaming validation
- 🔄 Reconnection logic testing
- 💓 Heartbeat monitoring

**Usage:**
```bash
# Test WebSocket functionality
python test_websocket_client.py

# Custom WebSocket URL
python test_websocket_client.py --url ws://your-domain.com/ws/ask
```

## 🚀 Running All E2E Tests

### Prerequisites
```bash
# Install additional dependencies for E2E tests
pip install aiohttp websockets
```

### Sequential Test Execution
```bash
cd backend/tests/e2e

# 1. Start your backend server
cd ../../
python -m uvicorn main:app --reload --port 8000

# 2. Run tests in another terminal
cd tests/e2e

# Deployment validation
python test_deployment.py

# Concurrency testing  
python test_thread_safety.py

# Error handling
python test_error_handling.py

# WebSocket functionality
python test_websocket_client.py
```

### Automated Test Suite
```bash
# Run all tests with a single script
./run_all_e2e_tests.sh  # (create this script)
```

## 📊 Test Reports and Results

### Generated Reports
- **`deployment_test_report.md`** - Deployment validation results
- **`thread_safety_test_results.json`** - Concurrency test metrics
- **Console output** - Real-time test progress and results

### Key Metrics Tracked
- **Response Times**: Average, min, max, median
- **Success Rates**: Percentage of successful requests
- **Concurrency**: Number of simultaneous users supported
- **Session Isolation**: Proper session management validation
- **Error Rates**: System stability under various conditions

## 🎯 CI/CD Integration

### GitHub Actions Example
```yaml
name: E2E Tests
on: [push, pull_request]

jobs:
  e2e-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          cd backend
          pip install -r requirements.txt
          pip install aiohttp websockets
      
      - name: Start backend
        run: |
          cd backend
          python -m uvicorn main:app --host 0.0.0.0 --port 8000 &
          sleep 10
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      
      - name: Run E2E tests
        run: |
          cd backend/tests/e2e
          python test_deployment.py --skip-docker
          python test_error_handling.py
```

## 🔧 Configuration

### Environment Variables for Testing
```bash
# Required for all tests
OPENAI_API_KEY=your_api_key

# Optional test configuration
TEST_BASE_URL=http://localhost:8000
TEST_TIMEOUT=30
TEST_CONCURRENCY=10
```

### Test Data
Tests use predefined DeFi-related questions:
- "What is DeFi?"
- "How does liquidity mining work?"
- "What are the risks of yield farming?"
- "How do automated market makers work?"
- And more...

## 🚨 Troubleshooting

### Common Issues

#### 1. Connection Refused
```bash
# Ensure backend is running
cd backend
python -m uvicorn main:app --reload
```

#### 2. OpenAI API Errors
```bash
# Check API key is set
echo $OPENAI_API_KEY

# Verify in .env file
cat .env | grep OPENAI_API_KEY
```

#### 3. Timeout Errors
```bash
# Increase timeout in test configuration
# Ensure system has sufficient resources
```

#### 4. Port Conflicts
```bash
# Check what's running on port 8000
lsof -i :8000

# Use different port for testing
python -m uvicorn main:app --port 8001
python test_deployment.py --url http://localhost:8001
```

## 📈 Performance Benchmarks

### Expected Performance (Production)
- **Response Time**: < 2 seconds average
- **Concurrent Users**: 50+ simultaneous connections
- **Success Rate**: > 99% for normal operations
- **Session Isolation**: 100% (no cross-contamination)
- **Memory Usage**: Stable under load

### Alerting Thresholds
- Response time > 5 seconds: ⚠️ Warning
- Success rate < 95%: 🚨 Critical
- Session isolation failures: 🚨 Critical
- Memory leaks detected: ⚠️ Warning

## 🤝 Contributing

### Adding New E2E Tests
1. Create test file following naming convention: `test_<feature>.py`
2. Include comprehensive error handling
3. Add progress logging and metrics collection
4. Update this README with test description
5. Add to CI/CD pipeline

### Test Guidelines
- **Comprehensive**: Cover happy path and edge cases
- **Isolated**: Tests should not depend on each other
- **Informative**: Clear progress logging and error messages
- **Configurable**: Support different environments and parameters
- **Documented**: Update README with new test information

---

**For unit tests, see the parent `tests/` directory**
**For general backend information, see `backend/README.md`** 