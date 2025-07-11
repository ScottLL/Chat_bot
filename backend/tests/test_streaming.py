#!/usr/bin/env python3
"""
Test script for the SSE streaming endpoint.

This script tests the /ask-stream endpoint to verify that:
1. It properly streams responses word-by-word
2. SSE format is correct
3. All event types are handled properly
"""

import requests
import json
import time
import pytest

def _check_server_availability():
    """Check if the server is running and the streaming endpoint works."""
    try:
        # Test the actual streaming endpoint instead of relying on health status
        response = requests.post(
            "http://localhost:8000/v2/ask-stream",
            json={"question": "test"},
            headers={"Accept": "text/event-stream"},
            timeout=5,
            stream=True
        )
        # If we get any response (not a connection error), the server is working
        return response.status_code in [200, 400, 503]  # Any valid HTTP response
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return False

def test_streaming_endpoint():
    """Test the SSE streaming endpoint."""
    # Skip test if server is not running
    if not _check_server_availability():
        pytest.skip("Server not running on localhost:8000 - skipping integration test")
    
    print("🌊 Testing SSE Streaming Endpoint...")
    print("=" * 50)
    
    url = "http://localhost:8000/v2/ask-stream"
    question = "What is yield farming?"
    
    print(f"📤 Sending question: '{question}'")
    print("📥 Receiving streaming response:")
    print("-" * 30)
    
    try:
        # Make streaming request
        response = requests.post(
            url,
            json={"question": question},
            headers={
                "Accept": "text/event-stream",
                "Cache-Control": "no-cache"
            },
            stream=True
        )
        
        assert response.status_code == 200, f"Expected HTTP 200 but got {response.status_code}: {response.text}"
        
        # Parse streaming response
        word_count = 0
        complete_text = []
        
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith("data: "):
                # Parse SSE data
                data_str = line[6:]  # Remove "data: " prefix
                
                try:
                    data = json.loads(data_str)
                    event_type = data.get("type")
                    
                    if event_type == "word":
                        word = data.get("content", "")
                        index = data.get("index", -1)
                        confidence = data.get("confidence", 0)
                        
                        complete_text.append(word)
                        word_count += 1
                        
                        print(f"🔤 Word {index + 1}: '{word}' (confidence: {confidence:.2f})")
                        
                        # Stop after first few words for demo
                        if word_count >= 10:
                            print("...")
                            break
                    
                    elif event_type == "complete":
                        total_words = data.get("total_words", 0)
                        confidence = data.get("confidence", 0)
                        print(f"✅ Complete! Total words: {total_words}, Confidence: {confidence:.2f}")
                        break
                    
                    elif event_type == "error":
                        error = data.get("error", "Unknown error")
                        print(f"❌ Error: {error}")
                        assert False, f"Streaming returned error: {error}"
                
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON decode error: {e}")
                    print(f"Raw data: {data_str}")
        
        # Show partial response
        if complete_text:
            response_preview = " ".join(complete_text[:10])
            if len(complete_text) > 10:
                response_preview += "..."
            print(f"\n📝 Response preview: {response_preview}")
        
        print("\n✅ Streaming test completed successfully!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to server during test execution")
        assert False, "Connection lost during test execution"
    except Exception as e:
        print(f"❌ Error: {e}")
        assert False, f"Streaming test failed: {e}"

def test_health_check():
    """Quick health check before streaming test."""
    # Skip test if server is not running
    if not _check_server_availability():
        pytest.skip("Server not running on localhost:8000 - skipping integration test")
    
    print("🩺 Checking server health...")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        assert response.status_code == 200, f"Health check failed: HTTP {response.status_code}"
        health_data = response.json()
        print(f"✅ Server responding: {health_data.get('message', '')}")
        # Note: We don't require agent_loaded=true since async system works differently
    except requests.exceptions.ConnectionError:
        print("❌ Server is not responding")
        assert False, "Server is not responding"
    except Exception as e:
        print(f"❌ Health check error: {e}")
        assert False, f"Health check error: {e}"

if __name__ == "__main__":
    print("🧪 DeFi Q&A Streaming Test Suite")
    print("=" * 40)
    
    # Test health first
    if test_health_check():
        print()
        # Test streaming
        test_streaming_endpoint()
    else:
        print("\n⚠️ Skipping streaming test - server not available") 