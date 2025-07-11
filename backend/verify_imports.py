#!/usr/bin/env python3
"""
Import verification script for the reorganized backend structure.

This script tests that all imports are working correctly after the reorganization.
"""

import sys
from pathlib import Path

def test_imports():
    """Test all the reorganized imports."""
    print("🔍 Testing reorganized imports...")
    
    try:
        # Test config import (should work - not moved)
        print("  ✓ Testing config import...")
        from config import config
        print(f"    ✅ Config loaded (environment: {config.ENVIRONMENT})")
        
        # Test agents imports individually
        print("  ✓ Testing agents imports...")
        
        print("    - Testing DeFiQAAgent...")
        from agents.defi_qa_agent import DeFiQAAgent
        print("      ✅ DeFiQAAgent imported")
        
        print("    - Testing AsyncDeFiQAAgent...")
        from agents.async_defi_qa_agent import AsyncDeFiQAAgent
        print("      ✅ AsyncDeFiQAAgent imported")
        
        print("    - Testing AsyncAgentFactory...")
        from agents.async_defi_qa_agent import AsyncAgentFactory
        print("      ✅ AsyncAgentFactory imported")
        
        print("    ✅ All agent imports successful")
        
        # Test services imports individually
        print("  ✓ Testing services imports...")
        
        print("    - Testing EmbeddingService...")
        from services.embedding_service import EmbeddingService
        print("      ✅ EmbeddingService imported")
        
        print("    - Testing AsyncEmbeddingService...")
        from services.async_embedding_service import AsyncEmbeddingService
        print("      ✅ AsyncEmbeddingService imported")
        
        print("    - Testing CacheManager...")
        from services.cache_manager import CacheManager
        print("      ✅ CacheManager imported")
        
        print("    - Testing QADataset...")
        from services.dataset_loader import QADataset
        print("      ✅ QADataset imported")
        
        print("    - Testing SessionManager and SessionContext...")
        from services.session_manager import SessionManager, SessionContext
        print("      ✅ SessionManager and SessionContext imported")
        
        print("    ✅ All services imports successful")
        
        # Test infrastructure imports
        print("  ✓ Testing infrastructure imports...")
        from infrastructure.websocket_manager import WebSocketConnectionManager
        from infrastructure.monitoring import MetricsCollector
        from infrastructure.error_handlers import ErrorHandler
        from infrastructure.logging_config import get_logger
        print("    ✅ Infrastructure imports successful")
        
        # Test that main.py can import everything
        print("  ✓ Testing main.py imports...")
        import main
        print("    ✅ Main application imports successful")
        
        print("\n🎉 All imports working correctly!")
        print("✅ Backend reorganization successful!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("💡 Check that all import statements have been updated correctly")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

def test_package_structure():
    """Test that the package structure is correct."""
    print("\n🏗️  Testing package structure...")
    
    required_files = [
        "agents/__init__.py",
        "services/__init__.py", 
        "infrastructure/__init__.py",
        "middleware/__init__.py",
        "tests/__init__.py",
        "docs/__init__.py",
        "agents/defi_qa_agent.py",
        "agents/async_defi_qa_agent.py",
        "services/embedding_service.py",
        "services/async_embedding_service.py",
        "services/cache_manager.py",
        "services/dataset_loader.py",
        "services/session_manager.py",
        "infrastructure/websocket_manager.py",
        "infrastructure/monitoring.py",
        "infrastructure/error_handlers.py",
        "infrastructure/logging_config.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    else:
        print("✅ All required files in correct locations")
        return True

if __name__ == "__main__":
    print("🔧 Backend Structure Verification")
    print("=" * 50)
    
    structure_ok = test_package_structure()
    imports_ok = test_imports()
    
    if structure_ok and imports_ok:
        print("\n🚀 Backend reorganization completed successfully!")
        print("📁 New structure:")
        print("   📦 agents/          - LangGraph agents")
        print("   📦 services/        - Business logic services")  
        print("   📦 infrastructure/  - Infrastructure components")
        print("   📦 middleware/      - FastAPI middleware")
        print("   📦 tests/          - Test suite")
        print("   📦 docs/           - Documentation")
        sys.exit(0)
    else:
        print("\n❌ Reorganization needs attention")
        sys.exit(1) 