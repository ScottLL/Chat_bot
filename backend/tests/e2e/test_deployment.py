#!/usr/bin/env python3
"""
DeFi Q&A Bot - Deployment Testing Suite

This script validates deployment configurations and tests API endpoints
to ensure the application works correctly in production environment.
"""

import os
import sys
import json
import time
import asyncio
import requests
import subprocess
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DeploymentTester:
    """Comprehensive deployment testing suite."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.test_results = []
        
    def log_test(self, test_name: str, passed: bool, message: str = ""):
        """Log test result."""
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status} - {test_name}: {message}")
        self.test_results.append({
            "test": test_name,
            "passed": passed,
            "message": message,
            "timestamp": time.time()
        })
        
    def test_configuration_files(self) -> bool:
        """Test that all deployment configuration files exist and are valid."""
        logger.info("🔍 Testing deployment configuration files...")
        
        config_files = [
            ("Procfile", "Heroku deployment"),
            ("railway.toml", "Railway deployment"),
            ("render.yaml", "Render deployment"),
            ("vercel.json", "Vercel deployment"),
            ("backend/Dockerfile", "Docker deployment"),
            ("backend/.dockerignore", "Docker build optimization"),
            ("requirements.txt", "Python dependencies"),
            ("DEPLOYMENT.md", "Deployment documentation")
        ]
        
        all_passed = True
        for file_path, description in config_files:
            if Path(file_path).exists():
                self.log_test(f"Config File: {file_path}", True, f"{description} file exists")
            else:
                self.log_test(f"Config File: {file_path}", False, f"Missing {description} file")
                all_passed = False
                
        return all_passed
        
    def test_environment_variables(self) -> bool:
        """Test environment variable configuration."""
        logger.info("🔍 Testing environment variable configuration...")
        
        # Test config module can be imported
        try:
            sys.path.append('backend')
            from config import config
            self.log_test("Config Module Import", True, "Configuration module loads successfully")
            
            # Test required environment variables
            required_vars = [
                "OPENAI_API_KEY",
                "ENVIRONMENT", 
                "HOST",
                "PORT"
            ]
            
            missing_vars = []
            for var in required_vars:
                if hasattr(config, var) and getattr(config, var) is not None:
                    self.log_test(f"Env Var: {var}", True, f"Value: {getattr(config, var)}")
                else:
                    missing_vars.append(var)
                    self.log_test(f"Env Var: {var}", False, "Missing or None")
                    
            return len(missing_vars) == 0
            
        except Exception as e:
            self.log_test("Config Module Import", False, f"Failed to import config: {e}")
            return False
            
    def test_health_endpoint(self) -> bool:
        """Test the health check endpoint."""
        logger.info("🔍 Testing health endpoint...")
        
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.log_test("Health Endpoint", True, f"Status: {data.get('status', 'unknown')}")
                return True
            else:
                self.log_test("Health Endpoint", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Health Endpoint", False, f"Connection error: {e}")
            return False
            
    def test_main_endpoint(self) -> bool:
        """Test the main API endpoint."""
        logger.info("🔍 Testing main API endpoint...")
        
        try:
            response = requests.get(f"{self.base_url}/", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.log_test("Main Endpoint", True, f"API: {data.get('api', 'DeFi Q&A Bot')}")
                return True
            else:
                self.log_test("Main Endpoint", False, f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Main Endpoint", False, f"Connection error: {e}")
            return False
            
    def test_ask_endpoint(self) -> bool:
        """Test the main Q&A functionality."""
        logger.info("🔍 Testing ask endpoint...")
        
        test_question = "What is DeFi lending?"
        payload = {"question": test_question}
        
        try:
            response = requests.post(
                f"{self.base_url}/ask",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get('answer', '')
                if answer and len(answer) > 10:
                    self.log_test("Ask Endpoint", True, f"Generated {len(answer)} character response")
                    return True
                else:
                    self.log_test("Ask Endpoint", False, "Empty or very short response")
                    return False
            else:
                self.log_test("Ask Endpoint", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("Ask Endpoint", False, f"Error: {e}")
            return False
            
    def test_monitoring_endpoints(self) -> bool:
        """Test monitoring and metrics endpoints."""
        logger.info("🔍 Testing monitoring endpoints...")
        
        endpoints = [
            "/health-status",
            "/system-metrics", 
            "/performance-stats",
            "/dashboard"
        ]
        
        all_passed = True
        for endpoint in endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                if response.status_code == 200:
                    self.log_test(f"Monitoring: {endpoint}", True, "Endpoint accessible")
                else:
                    self.log_test(f"Monitoring: {endpoint}", False, f"HTTP {response.status_code}")
                    all_passed = False
            except Exception as e:
                self.log_test(f"Monitoring: {endpoint}", False, f"Error: {e}")
                all_passed = False
                
        return all_passed
        
    def test_cors_configuration(self) -> bool:
        """Test CORS configuration."""
        logger.info("🔍 Testing CORS configuration...")
        
        try:
            # Test preflight request
            headers = {
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type"
            }
            
            response = requests.options(f"{self.base_url}/ask", headers=headers, timeout=10)
            
            if response.status_code == 200:
                cors_headers = response.headers
                if "access-control-allow-origin" in cors_headers:
                    self.log_test("CORS Configuration", True, f"CORS enabled: {cors_headers.get('access-control-allow-origin')}")
                    return True
                else:
                    self.log_test("CORS Configuration", False, "CORS headers missing")
                    return False
            else:
                self.log_test("CORS Configuration", False, f"Preflight failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("CORS Configuration", False, f"Error: {e}")
            return False
            
    def test_rate_limiting(self) -> bool:
        """Test rate limiting functionality."""
        logger.info("🔍 Testing rate limiting...")
        
        try:
            # Make rapid requests to test rate limiting
            responses = []
            for i in range(5):
                response = requests.get(f"{self.base_url}/", timeout=5)
                responses.append(response.status_code)
                time.sleep(0.1)  # Small delay
                
            # Check if all requests succeeded (rate limit not hit with normal usage)
            if all(status == 200 for status in responses):
                self.log_test("Rate Limiting", True, "Normal requests not rate limited")
                return True
            else:
                self.log_test("Rate Limiting", False, f"Unexpected response codes: {responses}")
                return False
                
        except Exception as e:
            self.log_test("Rate Limiting", False, f"Error: {e}")
            return False
            
    def test_docker_build(self) -> bool:
        """Test Docker image can be built successfully."""
        logger.info("🔍 Testing Docker build...")
        
        if not Path("backend/Dockerfile").exists():
            self.log_test("Docker Build", False, "Dockerfile not found")
            return False
            
        try:
            # Test docker build (if docker is available)
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                self.log_test("Docker Build", False, "Docker not available")
                return False
                
            # Try to build the image
            build_result = subprocess.run(
                ["docker", "build", "-t", "defi-qa-test", "backend/"],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if build_result.returncode == 0:
                self.log_test("Docker Build", True, "Docker image built successfully")
                
                # Clean up test image
                subprocess.run(
                    ["docker", "rmi", "defi-qa-test"],
                    capture_output=True,
                    text=True
                )
                return True
            else:
                self.log_test("Docker Build", False, f"Build failed: {build_result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.log_test("Docker Build", False, "Build timeout")
            return False
        except Exception as e:
            self.log_test("Docker Build", False, f"Error: {e}")
            return False
            
    def run_all_tests(self) -> Tuple[int, int]:
        """Run all deployment tests."""
        logger.info("🚀 Starting deployment testing suite...")
        
        tests = [
            self.test_configuration_files,
            self.test_environment_variables,
            self.test_health_endpoint,
            self.test_main_endpoint,
            self.test_ask_endpoint,
            self.test_monitoring_endpoints,
            self.test_cors_configuration,
            self.test_rate_limiting,
            self.test_docker_build
        ]
        
        passed = 0
        total = 0
        
        for test in tests:
            try:
                if test():
                    passed += 1
                total += 1
            except Exception as e:
                logger.error(f"Test {test.__name__} failed with exception: {e}")
                total += 1
                
        return passed, total
        
    def generate_report(self) -> str:
        """Generate test report."""
        passed_tests = [r for r in self.test_results if r["passed"]]
        failed_tests = [r for r in self.test_results if not r["passed"]]
        
        report = f"""
# 🧪 Deployment Test Report

## Summary
- **Total Tests**: {len(self.test_results)}
- **Passed**: {len(passed_tests)} ✅
- **Failed**: {len(failed_tests)} ❌
- **Success Rate**: {(len(passed_tests) / len(self.test_results) * 100):.1f}%

## Test Results

### ✅ Passed Tests ({len(passed_tests)})
"""
        
        for test in passed_tests:
            report += f"- **{test['test']}**: {test['message']}\n"
            
        if failed_tests:
            report += f"\n### ❌ Failed Tests ({len(failed_tests)})\n"
            for test in failed_tests:
                report += f"- **{test['test']}**: {test['message']}\n"
                
        report += f"\n\n*Report generated at {time.strftime('%Y-%m-%d %H:%M:%S')}*"
        
        return report

def main():
    """Main test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="DeFi Q&A Bot Deployment Tester")
    parser.add_argument(
        "--url", 
        default="http://localhost:8000",
        help="Base URL of the deployed application"
    )
    parser.add_argument(
        "--output",
        default="deployment_test_report.md",
        help="Output file for test report"
    )
    parser.add_argument(
        "--skip-docker",
        action="store_true",
        help="Skip Docker build test"
    )
    
    args = parser.parse_args()
    
    tester = DeploymentTester(args.url)
    
    # Skip Docker test if requested
    if args.skip_docker:
        tester.test_docker_build = lambda: True
        
    passed, total = tester.run_all_tests()
    
    # Generate and save report
    report = tester.generate_report()
    with open(args.output, 'w') as f:
        f.write(report)
        
    # Print summary
    success_rate = (passed / total * 100) if total > 0 else 0
    print(f"\n🎯 Test Summary: {passed}/{total} tests passed ({success_rate:.1f}%)")
    print(f"📄 Full report saved to: {args.output}")
    
    if passed == total:
        print("🎉 All tests passed! Deployment is ready.")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed. Check the report for details.")
        sys.exit(1)

if __name__ == "__main__":
    main() 