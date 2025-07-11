#!/usr/bin/env python3
"""
API Test Suite for DeFi Q&A Bot

Tests the FastAPI endpoints to ensure they work correctly with
the LangGraph agent integration.
"""

import json
import time
import asyncio
import subprocess
import sys
from typing import Dict, Any

import requests
from requests.exceptions import RequestException


class APITester:
    """Test suite for the DeFi Q&A API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the API tester."""
        self.base_url = base_url
        self.session = requests.Session()
        
    def test_health_endpoint(self) -> bool:
        """Test the health check endpoint."""
        print("🩺 Testing health endpoint...")
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Health check passed")
                print(f"   📊 Status: {data.get('status')}")
                print(f"   🤖 Agent loaded: {data.get('agent_loaded')}")
                print(f"   💬 Message: {data.get('message')}")
                return data.get('agent_loaded', False)
            else:
                print(f"   ❌ Health check failed: {response.status_code}")
                return False
                
        except RequestException as e:
            print(f"   ❌ Health check error: {e}")
            return False
    
    def test_root_endpoint(self) -> bool:
        """Test the root endpoint."""
        print("\n🏠 Testing root endpoint...")
        
        try:
            response = self.session.get(f"{self.base_url}/")
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Root endpoint working")
                print(f"   📝 Message: {data.get('message')}")
                return True
            else:
                print(f"   ❌ Root endpoint failed: {response.status_code}")
                return False
                
        except RequestException as e:
            print(f"   ❌ Root endpoint error: {e}")
            return False
    
    def test_ask_endpoint(self, question: str) -> Dict[str, Any]:
        """Test the /ask endpoint with a specific question."""
        print(f"\n❓ Testing /ask endpoint with: '{question}'")
        
        try:
            payload = {"question": question}
            response = self.session.post(
                f"{self.base_url}/ask",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Question processed successfully")
                print(f"   🤖 Answer: {data.get('answer', 'No answer')[:100]}...")
                print(f"   🎯 Confidence: {data.get('confidence', 0):.3f}")
                print(f"   📊 Stage: {data.get('processing_stage')}")
                
                if data.get('error'):
                    print(f"   ⚠️  Error: {data.get('error')}")
                
                return data
            else:
                print(f"   ❌ Ask endpoint failed: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   🔍 Error details: {error_data}")
                except:
                    print(f"   🔍 Raw response: {response.text}")
                return {}
                
        except RequestException as e:
            print(f"   ❌ Ask endpoint error: {e}")
            return {}
    
    def test_invalid_requests(self) -> bool:
        """Test invalid request handling."""
        print(f"\n🚫 Testing invalid request handling...")
        
        # Test empty question
        result1 = self.test_invalid_question("")
        
        # Test very short question
        result2 = self.test_invalid_question("Hi")
        
        # Test very long question
        long_question = "What is DeFi? " * 100  # Make it very long
        result3 = self.test_invalid_question(long_question)
        
        return result1 and result2 and result3
    
    def test_invalid_question(self, question: str) -> bool:
        """Test an invalid question."""
        try:
            payload = {"question": question}
            response = self.session.post(
                f"{self.base_url}/ask",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            # Should return 422 for validation errors or handle gracefully
            if response.status_code in [422, 400, 200]:
                print(f"   ✅ Invalid question handled correctly: {response.status_code}")
                return True
            else:
                print(f"   ❌ Unexpected status code: {response.status_code}")
                return False
                
        except RequestException as e:
            print(f"   ❌ Invalid request test error: {e}")
            return False
    
    def run_comprehensive_test(self) -> bool:
        """Run the complete test suite."""
        print("🧪 Starting Comprehensive API Test Suite")
        print("=" * 60)
        
        # Test 1: Health check
        agent_loaded = self.test_health_endpoint()
        if not agent_loaded:
            print("\n❌ Agent not loaded - some tests may fail")
        
        # Test 2: Root endpoint
        root_ok = self.test_root_endpoint()
        
        # Test 3: Valid DeFi questions
        test_questions = [
            "What is the largest lending pool on Aave?",
            "How does Uniswap V3 work?",
            "What are the risks of yield farming?",
            "What is liquidity mining?",
            "How do automated market makers work?"
        ]
        
        question_results = []
        for question in test_questions:
            result = self.test_ask_endpoint(question)
            question_results.append(bool(result.get('answer')))
        
        # Test 4: Invalid requests
        invalid_ok = self.test_invalid_requests()
        
        # Test 5: Edge case - non-DeFi question
        edge_result = self.test_ask_endpoint("What is the weather like today?")
        
        # Summary
        print(f"\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        print(f"   Health Check: {'✅ PASS' if agent_loaded else '❌ FAIL'}")
        print(f"   Root Endpoint: {'✅ PASS' if root_ok else '❌ FAIL'}")
        print(f"   Valid Questions: {sum(question_results)}/{len(question_results)} passed")
        print(f"   Invalid Requests: {'✅ PASS' if invalid_ok else '❌ FAIL'}")
        print(f"   Edge Cases: {'✅ PASS' if edge_result else '❌ FAIL'}")
        
        overall_success = (
            agent_loaded and 
            root_ok and 
            sum(question_results) >= len(question_results) * 0.8 and  # 80% success rate
            invalid_ok
        )
        
        print(f"\n🎯 OVERALL RESULT: {'✅ SUCCESS' if overall_success else '❌ FAILED'}")
        
        if overall_success:
            print("🚀 API is ready for frontend integration!")
        else:
            print("🔧 API needs debugging before frontend integration")
        
        return overall_success


def check_server_running(url: str = "http://localhost:8000") -> bool:
    """Check if the API server is running."""
    try:
        response = requests.get(f"{url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def main():
    """Main test function."""
    print("🚀 DeFi Q&A API Test Suite")
    print("=" * 60)
    
    # Check if server is running
    if not check_server_running():
        print("⚠️  API server is not running!")
        print("Please start the server first:")
        print("   cd backend")
        print("   python main.py")
        print("\nAlternatively, run:")
        print("   uvicorn main:app --reload")
        return False
    
    # Run tests
    tester = APITester()
    success = tester.run_comprehensive_test()
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 