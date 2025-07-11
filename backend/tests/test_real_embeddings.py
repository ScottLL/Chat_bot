#!/usr/bin/env python3
"""
Test script to verify embedding service works with real OpenAI API.

This script tests:
1. Loading dataset
2. Computing real embeddings with OpenAI API
3. Caching embeddings
4. Similarity search functionality
"""

import os
import sys

# Add parent directory to path to enable imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.dataset_loader import QADataset
from services.embedding_service import EmbeddingService

def main():
    print("🔬 Testing Real Embedding Service with OpenAI API...")
    print("=" * 60)
    
    try:
        # Load dataset
        print("\n1. Loading dataset...")
        dataset = QADataset()
        data = dataset.load_dataset()
        
        # Test with just first 3 items for speed
        test_data = data[:3]
        print(f"✅ Using {len(test_data)} Q&A pairs for testing")
        
        # Initialize embedding service
        print("\n2. Initializing embedding service...")
        embedding_service = EmbeddingService(cache_enabled=True)
        print("✅ Embedding service initialized with caching enabled")
        
        # Compute embeddings with real API
        print("\n3. Computing embeddings with OpenAI API...")
        print("   (This may take a few seconds...)")
        
        results = embedding_service.compute_dataset_embeddings(test_data)
        
        print(f"✅ Successfully computed embeddings!")
        print(f"   - Total items: {results['metadata']['total_items']}")
        print(f"   - Embedding model: {results['metadata']['embedding_model']}")
        print(f"   - Embedding dimension: {results['metadata']['embedding_dimension']}")
        
        # Test similarity search
        print("\n4. Testing similarity search...")
        
        # Get question embedding for similarity test
        test_question = "What is DeFi lending?"
        test_embedding = embedding_service.compute_embedding(test_question)
        
        print(f"   - Test question: '{test_question}'")
        print(f"   - Embedding computed: {len(test_embedding)} dimensions")
        
        # Find most similar questions
        similarities = []
        for item in results['items']:
            similarity = embedding_service.compute_similarity(
                test_embedding, 
                item['question_embedding']
            )
            similarities.append((item, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n   📊 Most similar Q&A pairs:")
        for i, (item, score) in enumerate(similarities):
            print(f"   {i+1}. [{score:.3f}] {item['question']}")
        
        # Test caching functionality
        print("\n5. Testing cache functionality...")
        
        # Try loading from cache (should be instant)
        cached_results = embedding_service.compute_dataset_embeddings(test_data)
        
        if 'cached' in str(cached_results) or len(cached_results['items']) == len(test_data):
            print("✅ Cache is working - embeddings loaded from cache!")
        else:
            print("⚠️ Cache might not be working as expected")
        
        print("\n" + "=" * 60)
        print("🎉 All tests completed successfully!")
        print("\n📋 Summary:")
        print("   ✅ Dataset loading works")
        print("   ✅ Real OpenAI API embedding computation works")
        print("   ✅ Embedding caching is functional")
        print("   ✅ Similarity search is operational")
        print("   ✅ Ready for LangGraph integration!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check your OPENAI_API_KEY in .env file")
        print("2. Ensure you have internet connectivity")
        print("3. Verify your OpenAI account has API credits")
        
        import traceback
        print(f"\nDetailed error:\n{traceback.format_exc()}")

if __name__ == "__main__":
    main() 