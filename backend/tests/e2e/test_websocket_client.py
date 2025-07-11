"""
WebSocket Test Client for DeFi Q&A Streaming

This script tests the WebSocket implementation by connecting to the server
and sending a test question to verify streaming functionality.
"""

import asyncio
import websockets
import json
import time
from typing import Dict, Any


class WebSocketTestClient:
    """Simple WebSocket test client for DeFi Q&A streaming."""
    
    def __init__(self, url: str = "ws://localhost:8000/ws/ask"):
        self.url = url
        self.websocket = None
        self.session_id = None
        self.messages_received = []
        self.words_received = []
        
    async def connect(self):
        """Connect to the WebSocket server."""
        try:
            print(f"🔗 Connecting to {self.url}...")
            self.websocket = await websockets.connect(self.url)
            print("✅ Connected successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from the WebSocket server."""
        if self.websocket:
            await self.websocket.close()
            print("🔌 Disconnected from WebSocket")
    
    async def send_message(self, message: Dict[str, Any]):
        """Send a message to the WebSocket server."""
        if not self.websocket:
            print("❌ Not connected to WebSocket")
            return
        
        try:
            await self.websocket.send(json.dumps(message))
            print(f"📤 Sent: {message}")
        except Exception as e:
            print(f"❌ Failed to send message: {e}")
    
    async def listen_for_messages(self, timeout: int = 30):
        """Listen for messages from the WebSocket server."""
        if not self.websocket:
            print("❌ Not connected to WebSocket")
            return
        
        print(f"👂 Listening for messages (timeout: {timeout}s)...")
        start_time = time.time()
        
        try:
            while time.time() - start_time < timeout:
                try:
                    # Wait for message with a short timeout
                    message = await asyncio.wait_for(
                        self.websocket.recv(), 
                        timeout=1.0
                    )
                    
                    try:
                        data = json.loads(message)
                        self.messages_received.append(data)
                        await self.handle_message(data)
                        
                        # Check for completion
                        if data.get("type") == "complete":
                            print("✅ Streaming completed!")
                            break
                        elif data.get("type") == "error":
                            print(f"❌ Error received: {data.get('error')}")
                            break
                            
                    except json.JSONDecodeError:
                        print(f"❌ Invalid JSON received: {message}")
                        
                except asyncio.TimeoutError:
                    # Continue listening, just print a heartbeat
                    elapsed = time.time() - start_time
                    if elapsed % 5 < 1:  # Print every 5 seconds
                        print(f"⏱️ Still listening... ({elapsed:.1f}s elapsed)")
                    continue
                    
        except Exception as e:
            print(f"❌ Error listening for messages: {e}")
    
    async def handle_message(self, data: Dict[str, Any]):
        """Handle incoming WebSocket messages."""
        message_type = data.get("type")
        timestamp = data.get("timestamp", time.time())
        
        if message_type == "status":
            status = data.get("status")
            details = data.get("details", {})
            print(f"🔄 Status: {status}")
            if status == "connected":
                self.session_id = details.get("session_id")
                print(f"🆔 Session ID: {self.session_id}")
                capabilities = details.get("capabilities", [])
                print(f"⚡ Capabilities: {', '.join(capabilities)}")
        
        elif message_type == "metadata":
            confidence = data.get("confidence", 0.0)
            total_words = data.get("total_words", 0)
            session_id = data.get("session_id")
            print(f"📊 Metadata: confidence={confidence:.2f}, total_words={total_words}, session={session_id}")
        
        elif message_type == "word":
            content = data.get("content", "")
            index = data.get("index", 0)
            confidence = data.get("confidence", 0.0)
            self.words_received.append(content)
            print(f"💬 Word {index}: '{content}' (confidence: {confidence:.2f})")
        
        elif message_type == "complete":
            processing_stage = data.get("processing_stage", "")
            final_confidence = data.get("final_confidence", 0.0)
            total_words = data.get("total_words", 0)
            print(f"✅ Complete: stage={processing_stage}, confidence={final_confidence:.2f}, words={total_words}")
            print(f"📝 Full response: {' '.join(self.words_received)}")
        
        elif message_type == "error":
            error = data.get("error", "Unknown error")
            processing_stage = data.get("processing_stage", "error")
            print(f"❌ Error: {error} (stage: {processing_stage})")
        
        elif message_type == "heartbeat":
            print(f"💓 Heartbeat received at {timestamp}")
        
        else:
            print(f"❓ Unknown message type: {message_type}")
            print(f"   Data: {data}")
    
    async def ask_question(self, question: str):
        """Send a question and listen for the streaming response."""
        if not self.websocket:
            print("❌ Not connected to WebSocket")
            return
        
        print(f"\n🤖 Asking question: '{question}'")
        
        # Send question message
        question_message = {
            "type": "question",
            "data": {
                "question": question
            }
        }
        
        await self.send_message(question_message)
        
        # Listen for response
        await self.listen_for_messages()
    
    async def test_basic_functionality(self):
        """Test basic WebSocket functionality."""
        print("🧪 Testing basic WebSocket functionality...")
        
        # Connect
        if not await self.connect():
            return False
        
        try:
            # Wait for initial connection status
            await asyncio.sleep(1)
            await self.listen_for_messages(timeout=3)
            
            # Ask a test question
            await self.ask_question("What is DeFi?")
            
            # Print summary
            print(f"\n📊 Test Summary:")
            print(f"   Messages received: {len(self.messages_received)}")
            print(f"   Words received: {len(self.words_received)}")
            print(f"   Session ID: {self.session_id}")
            
            return True
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
        
        finally:
            await self.disconnect()


async def main():
    """Run WebSocket tests."""
    print("🚀 Starting WebSocket Test Client")
    print("=" * 50)
    
    client = WebSocketTestClient()
    
    try:
        success = await client.test_basic_functionality()
        if success:
            print("\n✅ WebSocket test completed successfully!")
        else:
            print("\n❌ WebSocket test failed!")
    
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 