"""
Thread Safety and Concurrency Testing Suite

This module tests the thread safety and session consistency of our async DeFi Q&A system
under concurrent load to ensure proper isolation and no race conditions.
"""

import asyncio
import time
import random
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
import aiohttp
import threading
from collections import defaultdict, Counter
import statistics

class ThreadSafetyTester:
    """Comprehensive thread safety and concurrency tester for the DeFi Q&A system."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []
        self.session_tracking = defaultdict(list)
        self.timing_data = []
        self.error_counts = Counter()
        
    async def test_concurrent_requests(self, num_requests: int = 50, concurrency: int = 10) -> Dict[str, Any]:
        """Test multiple concurrent requests to validate thread safety."""
        print(f"\n🧪 Testing {num_requests} concurrent requests with {concurrency} concurrent connections...")
        
        # Test questions for variety
        test_questions = [
            "What is DeFi?",
            "How does liquidity mining work?",
            "What are the risks of yield farming?",
            "How do automated market makers work?",
            "What is impermanent loss?",
            "How do lending protocols work?",
            "What are governance tokens?",
            "How does flash lending work?",
            "What are wrapped tokens?",
            "How do decentralized exchanges work?"
        ]
        
        async with aiohttp.ClientSession() as session:
            # Create semaphore to control concurrency
            semaphore = asyncio.Semaphore(concurrency)
            
            async def make_request(question_id: int) -> Dict[str, Any]:
                async with semaphore:
                    question = random.choice(test_questions)
                    start_time = time.time()
                    
                    try:
                        async with session.post(
                            f"{self.base_url}/v2/ask",
                            json={"question": question},
                            timeout=aiohttp.ClientTimeout(total=30)
                        ) as response:
                            end_time = time.time()
                            duration = end_time - start_time
                            
                            if response.status == 200:
                                data = await response.json()
                                return {
                                    "request_id": question_id,
                                    "question": question,
                                    "status": "success",
                                    "response_time": duration,
                                    "session_id": data.get("session_id"),
                                    "confidence": data.get("confidence"),
                                    "answer_length": len(data.get("answer", "")),
                                    "timestamp": start_time
                                }
                            else:
                                error_text = await response.text()
                                return {
                                    "request_id": question_id,
                                    "question": question,
                                    "status": "error",
                                    "error": f"HTTP {response.status}: {error_text}",
                                    "response_time": duration,
                                    "timestamp": start_time
                                }
                    except Exception as e:
                        end_time = time.time()
                        return {
                            "request_id": question_id,
                            "question": question,
                            "status": "exception",
                            "error": str(e),
                            "response_time": end_time - start_time,
                            "timestamp": start_time
                        }
            
            # Execute all requests concurrently
            tasks = [make_request(i) for i in range(num_requests)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            return self._analyze_results(results)
    
    async def test_session_isolation(self, num_sessions: int = 20) -> Dict[str, Any]:
        """Test that sessions are properly isolated and don't interfere with each other."""
        print(f"\n🔒 Testing session isolation with {num_sessions} concurrent sessions...")
        
        async with aiohttp.ClientSession() as http_session:
            async def test_session_sequence(session_id: int) -> Dict[str, Any]:
                """Test a sequence of requests within a session to ensure isolation."""
                session_results = []
                session_questions = [
                    f"Session {session_id} question 1: What is DeFi?",
                    f"Session {session_id} question 2: How does yield farming work?",
                    f"Session {session_id} question 3: What are governance tokens?"
                ]
                
                for i, question in enumerate(session_questions):
                    try:
                        async with http_session.post(
                            f"{self.base_url}/v2/ask",
                            json={"question": question},
                            timeout=aiohttp.ClientTimeout(total=30)
                        ) as response:
                            if response.status == 200:
                                data = await response.json()
                                session_results.append({
                                    "session_id": session_id,
                                    "request_index": i,
                                    "question": question,
                                    "backend_session_id": data.get("session_id"),
                                    "confidence": data.get("confidence"),
                                    "status": "success"
                                })
                            else:
                                session_results.append({
                                    "session_id": session_id,
                                    "request_index": i,
                                    "question": question,
                                    "status": "error",
                                    "error": f"HTTP {response.status}"
                                })
                    except Exception as e:
                        session_results.append({
                            "session_id": session_id,
                            "request_index": i,
                            "question": question,
                            "status": "exception",
                            "error": str(e)
                        })
                
                return session_results
            
            # Run all sessions concurrently
            tasks = [test_session_sequence(i) for i in range(num_sessions)]
            all_session_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            return self._analyze_session_isolation(all_session_results)
    
    async def test_streaming_concurrency(self, num_streams: int = 15) -> Dict[str, Any]:
        """Test concurrent streaming requests for thread safety."""
        print(f"\n📡 Testing {num_streams} concurrent streaming requests...")
        
        async with aiohttp.ClientSession() as session:
            async def test_stream(stream_id: int) -> Dict[str, Any]:
                question = f"Stream {stream_id}: How do automated market makers work?"
                words_received = []
                start_time = time.time()
                
                try:
                    async with session.post(
                        f"{self.base_url}/v2/ask-stream",
                        json={"question": question},
                        timeout=aiohttp.ClientTimeout(total=60)
                    ) as response:
                        if response.status == 200:
                            async for line in response.content:
                                if line:
                                    try:
                                        data = json.loads(line.decode().strip())
                                        if data.get("type") == "word" and data.get("content"):
                                            words_received.append(data["content"])
                                    except json.JSONDecodeError:
                                        continue
                            
                            end_time = time.time()
                            return {
                                "stream_id": stream_id,
                                "status": "success",
                                "words_count": len(words_received),
                                "duration": end_time - start_time,
                                "avg_word_rate": len(words_received) / (end_time - start_time) if end_time > start_time else 0
                            }
                        else:
                            return {
                                "stream_id": stream_id,
                                "status": "error",
                                "error": f"HTTP {response.status}"
                            }
                except Exception as e:
                    return {
                        "stream_id": stream_id,
                        "status": "exception",
                        "error": str(e)
                    }
            
            # Run all streams concurrently
            tasks = [test_stream(i) for i in range(num_streams)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            return self._analyze_streaming_results(results)
    
    async def test_rate_limiting(self) -> Dict[str, Any]:
        """Test that rate limiting works correctly under load."""
        print(f"\n⚡ Testing rate limiting behavior...")
        
        # Send requests faster than the rate limit (50 req/sec)
        rapid_requests = 100
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            async def rapid_request(req_id: int) -> Dict[str, Any]:
                try:
                    async with session.post(
                        f"{self.base_url}/v2/ask",
                        json={"question": f"Rate limit test {req_id}"},
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        return {
                            "request_id": req_id,
                            "status_code": response.status,
                            "timestamp": time.time() - start_time
                        }
                except Exception as e:
                    return {
                        "request_id": req_id,
                        "error": str(e),
                        "timestamp": time.time() - start_time
                    }
            
            # Send all requests as fast as possible
            tasks = [rapid_request(i) for i in range(rapid_requests)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            return self._analyze_rate_limiting(results, start_time)
    
    def _analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the results of concurrent request testing."""
        successful = [r for r in results if isinstance(r, dict) and r.get("status") == "success"]
        errors = [r for r in results if isinstance(r, dict) and r.get("status") in ["error", "exception"]]
        exceptions = [r for r in results if not isinstance(r, dict)]
        
        response_times = [r["response_time"] for r in successful]
        session_ids = [r.get("session_id") for r in successful if r.get("session_id")]
        
        analysis = {
            "total_requests": len(results),
            "successful_requests": len(successful),
            "failed_requests": len(errors),
            "exceptions": len(exceptions),
            "success_rate": len(successful) / len(results) * 100 if results else 0,
            "unique_sessions": len(set(session_ids)) if session_ids else 0,
            "performance": {
                "avg_response_time": statistics.mean(response_times) if response_times else 0,
                "min_response_time": min(response_times) if response_times else 0,
                "max_response_time": max(response_times) if response_times else 0,
                "median_response_time": statistics.median(response_times) if response_times else 0
            }
        }
        
        if errors:
            error_types = Counter([r.get("error", "unknown") for r in errors])
            analysis["error_distribution"] = dict(error_types)
        
        return analysis
    
    def _analyze_session_isolation(self, all_session_results: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze session isolation test results."""
        total_sessions = len(all_session_results)
        successful_sessions = 0
        backend_session_consistency = 0
        cross_contamination_detected = False
        
        all_backend_sessions = set()
        
        for session_results in all_session_results:
            if isinstance(session_results, list):
                successful_requests = [r for r in session_results if r.get("status") == "success"]
                if len(successful_requests) == len(session_results):
                    successful_sessions += 1
                
                # Check if all requests in this session got the same backend session ID
                backend_ids = [r.get("backend_session_id") for r in successful_requests if r.get("backend_session_id")]
                if backend_ids and len(set(backend_ids)) == 1:
                    backend_session_consistency += 1
                    all_backend_sessions.add(backend_ids[0])
        
        # Check for session ID reuse across different test sessions
        if len(all_backend_sessions) < total_sessions:
            cross_contamination_detected = True
        
        return {
            "total_sessions_tested": total_sessions,
            "successful_sessions": successful_sessions,
            "session_success_rate": successful_sessions / total_sessions * 100 if total_sessions > 0 else 0,
            "backend_session_consistency": backend_session_consistency,
            "consistency_rate": backend_session_consistency / total_sessions * 100 if total_sessions > 0 else 0,
            "unique_backend_sessions": len(all_backend_sessions),
            "cross_contamination_detected": cross_contamination_detected,
            "isolation_score": "PASS" if not cross_contamination_detected and backend_session_consistency > 0 else "FAIL"
        }
    
    def _analyze_streaming_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze streaming concurrency test results."""
        successful = [r for r in results if isinstance(r, dict) and r.get("status") == "success"]
        failed = [r for r in results if isinstance(r, dict) and r.get("status") != "success"]
        
        word_counts = [r["words_count"] for r in successful]
        durations = [r["duration"] for r in successful]
        word_rates = [r["avg_word_rate"] for r in successful]
        
        return {
            "total_streams": len(results),
            "successful_streams": len(successful),
            "failed_streams": len(failed),
            "success_rate": len(successful) / len(results) * 100 if results else 0,
            "streaming_performance": {
                "avg_words_per_stream": statistics.mean(word_counts) if word_counts else 0,
                "avg_stream_duration": statistics.mean(durations) if durations else 0,
                "avg_word_rate": statistics.mean(word_rates) if word_rates else 0,
                "total_words_streamed": sum(word_counts)
            }
        }
    
    def _analyze_rate_limiting(self, results: List[Dict[str, Any]], start_time: float) -> Dict[str, Any]:
        """Analyze rate limiting test results."""
        successful = [r for r in results if isinstance(r, dict) and r.get("status_code") == 200]
        rate_limited = [r for r in results if isinstance(r, dict) and r.get("status_code") == 429]
        other_errors = [r for r in results if isinstance(r, dict) and r.get("status_code") not in [200, 429]]
        
        # Calculate actual request rate
        total_time = max([r.get("timestamp", 0) for r in results if isinstance(r, dict)])
        actual_rate = len(results) / total_time if total_time > 0 else 0
        
        return {
            "total_requests": len(results),
            "successful_requests": len(successful),
            "rate_limited_requests": len(rate_limited),
            "other_errors": len(other_errors),
            "actual_request_rate": actual_rate,
            "rate_limiting_working": len(rate_limited) > 0,
            "rate_limit_percentage": len(rate_limited) / len(results) * 100 if results else 0
        }

async def run_comprehensive_thread_safety_tests():
    """Run all thread safety and concurrency tests."""
    print("🚀 Starting Comprehensive Thread Safety and Concurrency Tests")
    print("=" * 70)
    
    tester = ThreadSafetyTester()
    all_results = {}
    
    try:
        # Test 1: Concurrent Request Handling
        all_results["concurrent_requests"] = await tester.test_concurrent_requests(
            num_requests=50, concurrency=10
        )
        
        # Test 2: Session Isolation
        all_results["session_isolation"] = await tester.test_session_isolation(num_sessions=20)
        
        # Test 3: Streaming Concurrency
        all_results["streaming_concurrency"] = await tester.test_streaming_concurrency(num_streams=15)
        
        # Test 4: Rate Limiting
        all_results["rate_limiting"] = await tester.test_rate_limiting()
        
        # Generate comprehensive report
        print("\n" + "=" * 70)
        print("📊 COMPREHENSIVE THREAD SAFETY TEST RESULTS")
        print("=" * 70)
        
        # Concurrent Requests Results
        cr = all_results["concurrent_requests"]
        print(f"\n🧪 CONCURRENT REQUESTS TEST:")
        print(f"   Total Requests: {cr['total_requests']}")
        print(f"   Success Rate: {cr['success_rate']:.1f}%")
        print(f"   Unique Sessions: {cr['unique_sessions']}")
        print(f"   Avg Response Time: {cr['performance']['avg_response_time']:.3f}s")
        print(f"   Min/Max Response Time: {cr['performance']['min_response_time']:.3f}s / {cr['performance']['max_response_time']:.3f}s")
        
        # Session Isolation Results
        si = all_results["session_isolation"]
        print(f"\n🔒 SESSION ISOLATION TEST:")
        print(f"   Sessions Tested: {si['total_sessions_tested']}")
        print(f"   Session Success Rate: {si['session_success_rate']:.1f}%")
        print(f"   Backend Session Consistency: {si['consistency_rate']:.1f}%")
        print(f"   Unique Backend Sessions: {si['unique_backend_sessions']}")
        print(f"   Cross-Contamination: {'❌ DETECTED' if si['cross_contamination_detected'] else '✅ NONE'}")
        print(f"   Isolation Score: {'✅ PASS' if si['isolation_score'] == 'PASS' else '❌ FAIL'}")
        
        # Streaming Concurrency Results
        sc = all_results["streaming_concurrency"]
        print(f"\n📡 STREAMING CONCURRENCY TEST:")
        print(f"   Total Streams: {sc['total_streams']}")
        print(f"   Success Rate: {sc['success_rate']:.1f}%")
        print(f"   Avg Words per Stream: {sc['streaming_performance']['avg_words_per_stream']:.1f}")
        print(f"   Avg Word Rate: {sc['streaming_performance']['avg_word_rate']:.1f} words/sec")
        print(f"   Total Words Streamed: {sc['streaming_performance']['total_words_streamed']}")
        
        # Rate Limiting Results
        rl = all_results["rate_limiting"]
        print(f"\n⚡ RATE LIMITING TEST:")
        print(f"   Total Requests: {rl['total_requests']}")
        print(f"   Successful: {rl['successful_requests']}")
        print(f"   Rate Limited: {rl['rate_limited_requests']}")
        print(f"   Actual Request Rate: {rl['actual_request_rate']:.1f} req/sec")
        print(f"   Rate Limiting Working: {'✅ YES' if rl['rate_limiting_working'] else '❌ NO'}")
        
        # Overall Assessment
        print(f"\n🎯 OVERALL THREAD SAFETY ASSESSMENT:")
        overall_success = (
            cr['success_rate'] > 95 and
            si['isolation_score'] == 'PASS' and
            sc['success_rate'] > 90 and
            rl['rate_limiting_working']
        )
        print(f"   Thread Safety Status: {'✅ EXCELLENT' if overall_success else '⚠️ NEEDS ATTENTION'}")
        
        # Save detailed results
        with open("thread_safety_test_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n📁 Detailed results saved to: thread_safety_test_results.json")
        print("=" * 70)
        
        return all_results
        
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    asyncio.run(run_comprehensive_thread_safety_tests()) 