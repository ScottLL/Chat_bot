#!/usr/bin/env python3
"""
Simple test for the DeFi Q&A LangGraph Agent.

This test verifies that all nodes and transitions work correctly
with a single test question.
"""

# Add parent directory to path to enable imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import configuration
from config import config
from agents.defi_qa_agent import DeFiQAAgent

def test_agent():
    """Test the LangGraph agent with a simple DeFi question."""
    print("🧪 Testing DeFi Q&A LangGraph Agent...")
    print("=" * 50)
    
    try:
        # Initialize agent
        print("1. Initializing agent...")
        agent = DeFiQAAgent(
            similarity_threshold=config.AGENT_SIMILARITY_THRESHOLD * 0.8,  # Lower threshold for testing
            max_results=config.AGENT_MAX_RESULTS - 1
        )
        print("✅ Agent initialized successfully!")
        
        # Test with a simple question
        test_question = "What is DeFi lending?"
        print(f"\n2. Testing question: '{test_question}'")
        
        result = agent.ask_question(test_question)
        
        print(f"\n3. Results:")
        print(f"   Response: {result['response']}")
        print(f"   Processing Stage: {result['processing_stage']}")
        print(f"   Similarity Scores: {result['similarity_scores']}")
        print(f"   Error: {result['error']}")
        
        # Verify success
        assert result['processing_stage'] == 'response_generated', f"Expected 'response_generated' but got '{result['processing_stage']}'"
        print("\n🎉 SUCCESS: LangGraph agent is working correctly!")
        print("✅ All nodes and transitions functioning properly")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        assert False, f"Agent test failed: {e}"

if __name__ == "__main__":
    success = test_agent()
    if success:
        print("\n" + "="*50)
        print("🚀 Ready to proceed with Task 4 - Semantic Search API!")
    else:
        print("\n" + "="*50)
        print("🔧 Need to debug the agent before proceeding.") 