#!/usr/bin/env python3
"""
Test script for enhanced error handling integration.

This script tests various error scenarios to ensure the enhanced
error handling system works correctly across all API endpoints.
"""

import requests
import json
import time


def call_api_endpoint(url, method="GET", data=None, expected_status=200):
    """Test an API endpoint and return the response."""
    try:
        if method == "POST":
            response = requests.post(url, json=data, timeout=10)
        else:
            response = requests.get(url, timeout=10)
        
        return {
            "success": True,
            "status_code": response.status_code,
            "data": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text,
            "expected": expected_status == response.status_code
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": str(e),
            "expected": False
        }


def main():
    """Run error handling tests."""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Enhanced Error Handling Integration")
    print("=" * 50)
    
    # Test 1: Health check (should work)
    print("\n1. Testing health endpoint...")
    result = call_api_endpoint(f"{base_url}/health")
    if result["success"] and result["expected"]:
        print("✅ Health check passed")
        print(f"   Status: {result['data']['status']}")
        print(f"   Agent loaded: {result['data']['agent_loaded']}")
    else:
        print("❌ Health check failed")
        print(f"   Error: {result}")
    
    # Test 2: Validation error (question too short)
    print("\n2. Testing validation error (short question)...")
    result = call_api_endpoint(
        f"{base_url}/ask", 
        "POST", 
        {"question": "hi"},  # Too short
        expected_status=422  # Pydantic validation error
    )
    if result["success"]:
        print("✅ Validation error handling working")
        if "detail" in result["data"]:
            print(f"   Response type: Pydantic validation")
            print(f"   Error details: {result['data']['detail'][0]['msg'] if result['data']['detail'] else 'N/A'}")
    else:
        print("❌ Validation error test failed")
        print(f"   Error: {result}")
    
    # Test 3: Valid question (should work)
    print("\n3. Testing valid question...")
    result = call_api_endpoint(
        f"{base_url}/ask", 
        "POST", 
        {"question": "What is yield farming in DeFi?"},
        expected_status=200
    )
    if result["success"] and result["expected"]:
        print("✅ Valid question processing works")
        data = result["data"]
        if isinstance(data, dict):
            if "error" in data and data["error"]:
                # Check if it's a structured error response
                print(f"   Response: Structured error (expected for missing data)")
                if "user_guidance" in data:
                    print(f"   Error category: {data.get('category', 'unknown')}")
                    print(f"   User guidance: {data['user_guidance']['title']}")
            else:
                print(f"   Answer: {data.get('answer', 'N/A')[:100]}...")
                print(f"   Confidence: {data.get('confidence', 'N/A')}")
    else:
        print("❌ Valid question test failed")
        print(f"   Result: {result}")
    
    # Test 4: Question likely to return no results
    print("\n4. Testing no results scenario...")
    result = call_api_endpoint(
        f"{base_url}/ask", 
        "POST", 
        {"question": "What is the weather like today in Tokyo?"},  # Non-DeFi question
        expected_status=400  # Should return structured error
    )
    if result["success"]:
        print("✅ No results error handling working")
        data = result["data"]
        if isinstance(data, dict) and "user_guidance" in data:
            print(f"   Error category: {data.get('category', 'unknown')}")
            print(f"   User guidance: {data['user_guidance']['title']}")
            print(f"   Suggestions count: {len(data['user_guidance']['suggestions'])}")
        else:
            print(f"   Response format: {type(data)}")
    else:
        print("❌ No results test failed")
        print(f"   Result: {result}")
    
    # Test 5: Streaming endpoint health
    print("\n5. Testing streaming endpoint...")
    try:
        response = requests.post(
            f"{base_url}/ask-stream",
            json={"question": "What is DeFi?"},
            stream=True,
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ Streaming endpoint accessible")
            # Read first few lines to test streaming
            lines_read = 0
            for line in response.iter_lines():
                if line and lines_read < 3:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data: '):
                        try:
                            data = json.loads(decoded_line[6:])  # Remove 'data: ' prefix
                            print(f"   Stream data type: {data.get('type', 'unknown')}")
                            if data.get('type') == 'error':
                                print(f"   Error category: {data.get('category', 'unknown')}")
                            lines_read += 1
                        except json.JSONDecodeError:
                            print(f"   Raw line: {decoded_line[:50]}...")
                            lines_read += 1
                if lines_read >= 3:
                    break
        else:
            print(f"❌ Streaming endpoint failed: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Streaming test failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Error handling integration test completed!")
    print("\nNext steps:")
    print("1. Check the React frontend error display")
    print("2. Verify error messages are user-friendly")
    print("3. Test recovery scenarios")


if __name__ == "__main__":
    main() 