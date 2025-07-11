# WebSocket Connection Testing Instructions

## Current Status

✅ **Services are running and ready for testing**
- Backend: `defi-qa-backend` - Started successfully
- Frontend: `defi-qa-frontend` - Built and serving
- New debugging features added to both frontend and backend

## How to Test the WebSocket Fix

### 1. Open the Application
1. Navigate to: **http://localhost**
2. Open browser Developer Console (Press **F12**)
3. Go to the **Console** tab

### 2. Test Basic WebSocket Functionality

#### Step 2.1: Test Simple WebSocket Connection
If the main WebSocket connection fails, you should see a **"Test WS"** button next to the "Reconnect" button.

1. Click the **"Test WS"** button
2. Watch the console for logs like:
   ```javascript
   🧪 Testing basic WebSocket connection...
   ✅ Test WebSocket connected successfully
   📨 Test WebSocket response: {type: "echo", message: "Received: {type: 'test', message: 'Hello from frontend'}"}
   🔌 Test WebSocket closed - Code: 1000, Reason: 
   ```

**If Test WS works**: Basic WebSocket functionality is OK, the issue is with the main endpoint dependencies.

**If Test WS fails**: There's a fundamental WebSocket/networking issue.

#### Step 2.2: Test Main WebSocket Connection
1. Refresh the page (F5 or Ctrl+R)
2. Watch the console for connection logs:

**Expected Success Logs:**
```javascript
🔗 Attempting WebSocket connection to: ws://localhost/ws/ask
✅ WebSocket connected successfully
📨 WebSocket message received: status
```

**Expected Failure Logs:**
```javascript
🔗 Attempting WebSocket connection to: ws://localhost/ws/ask
🔌 WebSocket disconnected - Code: 1006, Reason: 
⏱️ Reconnecting in 1000ms (attempt 1/5)...
```

### 3. Monitor Backend Logs

#### Step 3.1: Real-time Log Monitoring
Open a terminal and run:
```bash
docker logs defi-qa-backend -f
```

#### Step 3.2: Expected Backend Logs for Successful Connection
```
🔍 WebSocket connection attempt - Session: <session-id>
✅ WebSocket connected - Session: <session-id>
📤 Sent initial status message to session <session-id>
```

#### Step 3.3: Expected Backend Logs for Failed Connection
```
🔍 WebSocket connection attempt - Session: <session-id>
❌ WebSocket error - Session: <session-id>, Error: <error details>
<stack trace>
```

### 4. Test Scenarios

#### Scenario A: Fresh Page Load
1. Clear browser cache and cookies
2. Navigate to http://localhost
3. Observe initial connection attempt

#### Scenario B: Page Refresh
1. Once on the page, press F5 to refresh
2. Observe if connection establishes immediately
3. This was the original problem case

#### Scenario C: Network Interruption
1. Stop backend: `docker stop defi-qa-backend`
2. Observe automatic reconnection attempts
3. Restart backend: `docker start defi-qa-backend`
4. Verify connection re-establishes

#### Scenario D: Manual Reconnection
1. If connection fails, use the **"Reconnect"** button
2. Verify this triggers a new connection attempt

### 5. Status Indicators

**Connection Status Colors:**
- 🟢 **Green**: Connected successfully
- 🟡 **Yellow**: Connecting... (should be brief)
- 🔴 **Red**: Connection error
- ⚪ **Gray**: Disconnected

**Button States:**
- **"Ask Question"**: Available when connected
- **"Connecting..."**: Shown during connection attempt
- **"Reconnect"**: Appears when connection fails
- **"Test WS"**: Appears when connection fails (for debugging)

### 6. Common Issues and Solutions

#### Issue: HTTP 499 Errors in Nginx Logs
**Symptoms:** Frontend logs show connection attempts, but nginx logs show 499 status codes
**Likely Cause:** WebSocket handshake is failing due to dependency injection errors
**Solution:** Check backend logs for detailed error traces

#### Issue: Connection Hangs in "Connecting..." State
**Symptoms:** Yellow indicator stays for more than 5 seconds
**Likely Cause:** WebSocket endpoint is not responding
**Solution:** Check if backend is running and accessible

#### Issue: Immediate Disconnection (Code 1006)
**Symptoms:** Connection briefly succeeds then immediately closes
**Likely Cause:** Error during WebSocket message processing
**Solution:** Look for error traces in backend logs

#### Issue: Network/Docker Issues
**Symptoms:** Test WS also fails
**Likely Cause:** Docker networking or nginx proxy configuration
**Solution:** Verify docker-compose services and nginx.conf

### 7. Debugging Information to Collect

If issues persist, collect this information:

#### Frontend Console Logs
```javascript
// Look for these patterns:
🔗 Attempting WebSocket connection to: ...
✅ WebSocket connected successfully
🔌 WebSocket disconnected - Code: ...
❌ WebSocket error: ...
```

#### Backend Docker Logs
```bash
# Get recent logs
docker logs defi-qa-backend --tail 50

# Look for these patterns:
🔍 WebSocket connection attempt - Session: ...
✅ WebSocket connected - Session: ...
❌ WebSocket error - Session: ...
```

#### Nginx Access Logs
```bash
# Check frontend logs for nginx access patterns
docker logs defi-qa-frontend --tail 20

# Look for:
GET /ws/ask HTTP/1.1" 101  # Success
GET /ws/ask HTTP/1.1" 499  # Client closed (error)
```

#### Network Connectivity Test
```bash
# Test if backend is accessible
curl http://localhost:8000/health

# Expected response:
{"status": "healthy", "agent_loaded": true, ...}
```

### 8. Success Criteria

The WebSocket fix is working correctly when:

1. ✅ Page loads and connects immediately (green indicator)
2. ✅ Page refresh connects immediately without hanging
3. ✅ Manual reconnect button works when needed
4. ✅ Test WS button confirms basic WebSocket functionality
5. ✅ Backend logs show successful session creation and connection
6. ✅ No 499 errors in nginx logs
7. ✅ Able to ask questions and receive streaming responses

### 9. Rollback Plan

If the fix doesn't work and causes regressions:

```bash
# Revert to previous version
git revert HEAD

# Rebuild and restart
docker-compose down
docker-compose up --build -d
```

---

## Technical Details

### What Was Fixed

1. **Dependency Injection Issue**: Created specialized `get_websocket_session()` and `get_websocket_agent()` functions for WebSocket connections
2. **Session Lifecycle**: WebSocket sessions now persist for the entire connection duration instead of being cleaned up immediately after handshake
3. **Error Logging**: Added comprehensive debugging to identify connection issues
4. **Manual Recovery**: Added UI controls for manual reconnection and testing

### Files Modified

- `backend/main.py`: WebSocket dependency injection and debugging
- `frontend/src/App.tsx`: Connection logging and test functionality
- `WEBSOCKET_FIX.md`: Detailed technical documentation

---

**Start Testing Now:**
1. Open http://localhost in your browser
2. Open browser console (F12)
3. Follow the test scenarios above
4. Report results with console logs and backend logs 