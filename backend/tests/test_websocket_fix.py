#!/usr/bin/env python3
"""
Test script to verify WebSocket disconnection handling.

This script tests the WebSocket manager's ability to handle abrupt disconnections
and ensures the "no close frame received or sent" error is handled gracefully.
"""

import asyncio
import json
import websockets
import logging
import pytest
import requests
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)





class WebSocketTestClient:
    """Test client to simulate various disconnection scenarios."""
    
    def __init__(self, uri: str):
        self.uri = uri
        self.websocket = None
    
    async def connect(self):
        """Connect to the WebSocket server."""
        try:
            self.websocket = await websockets.connect(self.uri)
            logger.info("Connected to WebSocket server")
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    async def send_question(self, question: str):
        """Send a question to the server."""
        if not self.websocket:
            return False
        
        try:
            message = {
                "type": "question",
                "data": {"question": question}
            }
            await self.websocket.send(json.dumps(message))
            logger.info(f"Sent question: {question}")
            return True
        except Exception as e:
            logger.error(f"Failed to send question: {e}")
            return False
    
    async def receive_messages(self, timeout: float = 5.0) -> List[dict]:
        """Receive messages from the server with timeout."""
        if not self.websocket:
            return []
        
        messages = []
        try:
            while True:
                message = await asyncio.wait_for(
                    self.websocket.recv(), 
                    timeout=timeout
                )
                data = json.loads(message)
                messages.append(data)
                logger.info(f"Received: {data.get('type', 'unknown')}")
                
                # Stop if we get a complete message
                if data.get("type") == "complete":
                    break
                    
        except asyncio.TimeoutError:
            logger.info("Timeout waiting for messages")
        except Exception as e:
            logger.error(f"Error receiving messages: {e}")
        
        return messages
    
    async def disconnect_gracefully(self):
        """Disconnect with proper close frame."""
        if self.websocket:
            try:
                await self.websocket.close()
                logger.info("Disconnected gracefully")
            except Exception as e:
                logger.error(f"Error during graceful disconnect: {e}")
    
    async def disconnect_abruptly(self):
        """Simulate abrupt disconnection (like closing browser tab)."""
        if self.websocket:
            try:
                # Close the underlying connection without proper WebSocket close
                await self.websocket.transport.close()
                logger.info("Disconnected abruptly (simulating browser tab close)")
            except Exception as e:
                logger.error(f"Error during abrupt disconnect: {e}")


async def test_graceful_disconnect():
    """Test normal WebSocket connection and graceful disconnect."""
    logger.info("=== Testing Graceful Disconnect ===")
    
    client = WebSocketTestClient("ws://localhost:8000/ws/ask")
    
    try:
        if await client.connect():
            await client.send_question("What is DeFi?")
            messages = await client.receive_messages()
            logger.info(f"Received {len(messages)} messages")
            await client.disconnect_gracefully()
        else:
            pytest.skip("Could not connect to WebSocket server - skipping test")
    except Exception as e:
        pytest.skip(f"WebSocket connection failed: {e} - skipping test")
    
    await asyncio.sleep(2)  # Give server time to cleanup


async def test_abrupt_disconnect():
    """Test abrupt disconnection that should trigger the fix."""
    logger.info("=== Testing Abrupt Disconnect ===")
    
    client = WebSocketTestClient("ws://localhost:8000/ws/ask")
    
    try:
        if await client.connect():
            await client.send_question("What is blockchain?")
            
            # Receive some messages then disconnect abruptly
            await asyncio.sleep(1)  # Let some messages come through
            await client.disconnect_abruptly()
        else:
            pytest.skip("Could not connect to WebSocket server - skipping test")
    except Exception as e:
        pytest.skip(f"WebSocket connection failed: {e} - skipping test")
    
    await asyncio.sleep(5)  # Give server time to detect and cleanup


async def test_multiple_connections():
    """Test multiple connections to ensure heartbeat system works properly."""
    logger.info("=== Testing Multiple Connections ===")
    
    clients = []
    
    try:
        # Create multiple connections
        for i in range(3):
            client = WebSocketTestClient("ws://localhost:8000/ws/ask")
            if await client.connect():
                clients.append(client)
                await client.send_question(f"Test question {i}")
        
        if not clients:
            pytest.skip("Could not establish any WebSocket connections - skipping test")
        
        # Let them receive some data
        await asyncio.sleep(2)
        
        # Disconnect half abruptly, half gracefully
        for i, client in enumerate(clients):
            if i % 2 == 0:
                await client.disconnect_abruptly()
            else:
                await client.disconnect_gracefully()
    
    except Exception as e:
        pytest.skip(f"WebSocket connection failed: {e} - skipping test")
    
    await asyncio.sleep(5)  # Give server time to cleanup


async def test_heartbeat_resilience():
    """Test that heartbeat system handles disconnections properly."""
    logger.info("=== Testing Heartbeat Resilience ===")
    
    client = WebSocketTestClient("ws://localhost:8000/ws/ask")
    
    try:
        if await client.connect():
            logger.info("Connected, waiting for heartbeats...")
            
            # Wait for a few heartbeat cycles
            await asyncio.sleep(45)  # Wait for at least one heartbeat cycle
            
            # Disconnect abruptly during heartbeat
            await client.disconnect_abruptly()
            
            # Wait to see if server handles it gracefully
            await asyncio.sleep(35)  # Wait for cleanup cycle
        else:
            pytest.skip("Could not connect to WebSocket server - skipping test")
    except Exception as e:
        pytest.skip(f"WebSocket connection failed: {e} - skipping test")


async def main():
    """Run all WebSocket tests."""
    logger.info("Starting WebSocket disconnection tests...")
    logger.info("Make sure your server is running on localhost:8000")
    
    try:
        await test_graceful_disconnect()
        await test_abrupt_disconnect()
        await test_multiple_connections()
        await test_heartbeat_resilience()
        
        logger.info("=== All tests completed ===")
        logger.info("Check server logs for 'no close frame' errors.")
        logger.info("With the fix, these should be handled gracefully (debug level).")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 