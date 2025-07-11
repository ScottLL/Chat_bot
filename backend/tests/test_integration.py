#!/usr/bin/env python3
"""
Integration test for the complete dataset loading and embedding pipeline.

This script tests the full workflow:
1. Load Q&A dataset
2. Compute embeddings with caching
3. Verify cached embeddings can be loaded
4. Test similarity search functionality
"""

import os
import sys
import time

# Add parent directory to path to enable imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.dataset_loader import QADataset
from services.embedding_service import EmbeddingService
from services.cache_manager import CacheManager


def test_dataset_loading():
    """Test dataset loading functionality."""
    print("\n🔬 Testing Dataset Loading...")
    
    try:
        dataset = QADataset()
        data = dataset.load_dataset()
        
        print(f"✅ Dataset loaded successfully!")
        print(f"   - Total items: {len(data)}")
        print(f"   - First question: {data[0]['question'][:50]}...")
        
        stats = dataset.get_dataset_stats()
        print(f"   - Average question length: {stats['avg_question_length']:.1f} chars")
        print(f"   - Average answer length: {stats['avg_answer_length']:.1f} chars")
        
        assert data is not None and len(data) > 0, "Dataset should be loaded successfully"
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        assert False, f"Dataset loading failed: {e}"


def test_cache_manager():
    """Test cache manager basic functionality."""
    print("\n🔬 Testing Cache Manager...")
    
    try:
        cache = CacheManager(cache_dir="../cache_test")
        
        # Test embedding cache
        test_embeddings = [[0.1, 0.2], [0.3, 0.4]]
        cache_key = "test_integration_embeddings"
        
        # Save embeddings
        save_result = cache.save_embeddings(cache_key, test_embeddings, {"test": True})
        assert save_result, "Should be able to save test embeddings to cache"
        print("✅ Test embeddings saved to cache")
            
        # Load embeddings
        loaded = cache.load_embeddings(cache_key)
        assert loaded is not None, "Should be able to load embeddings from cache"
        assert loaded["embeddings"] == test_embeddings, "Loaded embeddings should match saved embeddings"
        print("✅ Test embeddings loaded from cache successfully")
            
        # Clean up test cache
        cache.clear_cache()
        print("✅ Test cache cleaned up")
        
    except Exception as e:
        print(f"❌ Cache manager test failed: {e}")
        assert False, f"Cache manager test failed: {e}"


def test_embedding_service_without_api():
    """Test embedding service initialization and basic methods without API calls."""
    print("\n🔬 Testing Embedding Service (No API)...")
    
    try:
        # Test with cache enabled but no API key (should fail gracefully)
        try:
            service = EmbeddingService(api_key="fake_key_for_testing")
            print("✅ Embedding service initialized")
            
            # Test similarity calculation
            emb1 = [1.0, 0.0, 0.0]
            emb2 = [0.0, 1.0, 0.0]
            emb3 = [1.0, 0.0, 0.0]  # Same as emb1
            
            sim_different = service.compute_similarity(emb1, emb2)
            sim_same = service.compute_similarity(emb1, emb3)
            
            print(f"✅ Similarity calculations work:")
            print(f"   - Different vectors: {sim_different:.3f}")
            print(f"   - Same vectors: {sim_same:.3f}")
            
            assert abs(sim_same - 1.0) < 0.001, "Similarity calculation should be accurate for identical vectors"
            print("✅ Similarity calculation is accurate")
                
        except ValueError as e:
            if "API key" in str(e):
                print("✅ API key validation works correctly")
            else:
                assert False, f"Unexpected error: {e}"
                
    except Exception as e:
        print(f"❌ Embedding service test failed: {e}")
        assert False, f"Embedding service test failed: {e}"


def test_complete_workflow_simulation():
    """Simulate the complete workflow without making actual API calls."""
    print("\n🔬 Testing Complete Workflow (Simulation)...")
    
    try:
        # Load dataset
        dataset = QADataset()
        data = dataset.load_dataset()
        
        assert data is not None and len(data) > 0, "Failed to load dataset for workflow test"
            
        print(f"✅ Loaded {len(data)} Q&A pairs for workflow test")
        
        # Simulate embedding computation results
        simulated_results = {
            'items': [],
            'metadata': {
                'total_items': len(data),
                'embedding_model': 'text-embedding-3-small',
                'embedding_dimension': 1536,
                'computed_at': time.time()
            }
        }
        
        # Create simulated embeddings (random but consistent)
        for i, item in enumerate(data):
            # Use hash to create consistent "fake" embeddings
            q_hash = hash(item['question']) % 1000
            a_hash = hash(item['answer']) % 1000
            
            simulated_results['items'].append({
                'id': item['id'],
                'question': item['question'],
                'answer': item['answer'],
                'question_embedding': [float(q_hash + j) / 1000 for j in range(5)],  # 5D for testing
                'answer_embedding': [float(a_hash + j) / 1000 for j in range(5)]
            })
        
        print(f"✅ Simulated embedding computation for {len(data)} items")
        
        # Test caching the simulated results
        cache = CacheManager(cache_dir="../cache_test")
        cache_key = "simulated_embeddings_test"
        
        save_success = cache.save_embeddings(cache_key, simulated_results['items'], simulated_results['metadata'])
        assert save_success, "Failed to save simulated embeddings"
        print("✅ Simulated embeddings saved to cache")
            
        # Load and verify
        loaded = cache.load_embeddings(cache_key)
        assert loaded is not None, "Failed to load simulated embeddings"
        assert len(loaded['embeddings']) == len(data), "Loaded embeddings count should match data count"
        print("✅ Simulated embeddings loaded from cache successfully")
        print(f"   - Loaded {len(loaded['embeddings'])} items")
        print(f"   - Embedding dimension: {loaded['metadata'].get('embedding_dimension', 'unknown')}")
            
        # Clean up test cache
        cache.clear_cache()
        print("✅ Test cache cleaned up")
        
    except Exception as e:
        print(f"❌ Complete workflow test failed: {e}")
        assert False, f"Complete workflow test failed: {e}"


def main():
    """Run all integration tests."""
    print("🚀 Starting Integration Tests...")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Test 1: Dataset Loading
    if not test_dataset_loading():
        all_tests_passed = False
        
    # Test 2: Cache Manager
    if not test_cache_manager():
        all_tests_passed = False
        
    # Test 3: Embedding Service (without API)
    if not test_embedding_service_without_api():
        all_tests_passed = False
        
    # Test 4: Complete Workflow Simulation
    if not test_complete_workflow_simulation():
        all_tests_passed = False
    
    print("\n" + "=" * 50)
    
    if all_tests_passed:
        print("🎉 All integration tests PASSED!")
        print("\n📋 Summary:")
        print("   ✅ Dataset loading works correctly")
        print("   ✅ Cache manager saves and loads embeddings")
        print("   ✅ Embedding service initializes and computes similarity")
        print("   ✅ Complete workflow simulation successful")
        print("\n🔑 Ready for API key configuration and real embedding computation!")
    else:
        print("❌ Some integration tests FAILED!")
        print("   Please check the errors above and fix any issues.")
        sys.exit(1)


if __name__ == "__main__":
    main() 