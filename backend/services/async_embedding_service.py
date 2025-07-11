"""
Async embedding service for computing and managing text embeddings.

This module provides non-blocking functionality to compute embeddings using 
OpenAI's AsyncOpenAI client for improved concurrent request handling.
"""

import os
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import openai
from openai import AsyncOpenAI
import numpy as np
# Import configuration
from config import config
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class AsyncEmbeddingService:
    """Async service for computing and managing text embeddings."""
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: str = None,
        batch_size: int = 20,  # Reduced for better rate limiting
        max_retries: int = 3,
        timeout: int = 30,
        max_concurrent: int = None  # Limit concurrent API calls
    ):
        """
        Initialize the async embedding service.
        
        Args:
            api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
            model: Embedding model to use 
            batch_size: Number of texts to process in each batch
            max_retries: Maximum number of API retry attempts
            timeout: API request timeout in seconds
            max_concurrent: Maximum concurrent API calls
        """
        # Use config values when not explicitly provided
        self.api_key = api_key or config.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided or set in OPENAI_API_KEY environment variable"
            )
            
        self.model = model or config.AGENT_EMBEDDING_MODEL
        max_concurrent = max_concurrent or config.MAX_CONCURRENT_REQUESTS
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Initialize AsyncOpenAI client
        self.client = AsyncOpenAI(api_key=self.api_key)
        
        # Concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent)
        
        # Thread-safe in-memory cache
        self._cache_lock = asyncio.Lock()
        self.embeddings_cache: Dict[str, List[float]] = {}
        
    async def compute_embedding(self, text: str) -> List[float]:
        """
        Compute embedding for a single text asynchronously.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
            
        Raises:
            Exception: If API call fails after all retries
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
            
        # Check cache first
        text_hash = str(hash(text))
        async with self._cache_lock:
            if text_hash in self.embeddings_cache:
                return self.embeddings_cache[text_hash]
        
        # Use semaphore to limit concurrent API calls
        async with self._semaphore:
            for attempt in range(self.max_retries):
                try:
                    response = await self.client.embeddings.create(
                        model=self.model,
                        input=text.strip(),
                        timeout=self.timeout
                    )
                    
                    embedding = response.data[0].embedding
                    
                    # Cache the result
                    async with self._cache_lock:
                        self.embeddings_cache[text_hash] = embedding
                    
                    return embedding
                    
                except openai.RateLimitError as e:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}")
                    await asyncio.sleep(wait_time)
                    
                except openai.APIError as e:
                    print(f"OpenAI API error on attempt {attempt + 1}: {e}")
                    if attempt == self.max_retries - 1:
                        raise
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    print(f"Unexpected error on attempt {attempt + 1}: {e}")
                    if attempt == self.max_retries - 1:
                        raise
                    await asyncio.sleep(1)
        
        raise Exception(f"Failed to compute embedding after {self.max_retries} attempts")
    
    async def compute_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Compute embeddings for a batch of texts concurrently.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding lists
        """
        if not texts:
            return []
        
        # Create tasks for concurrent processing
        tasks = []
        for text in texts:
            task = asyncio.create_task(self._safe_compute_embedding(text))
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        embeddings = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Failed to compute embedding for text {i}: {result}")
                # Use zero vector as fallback
                embeddings.append([0.0] * 1536)  # text-embedding-3-small dimension
            else:
                embeddings.append(result)
        
        return embeddings
    
    async def _safe_compute_embedding(self, text: str) -> List[float]:
        """
        Safely compute embedding with error handling.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or zero vector on error
        """
        try:
            return await self.compute_embedding(text)
        except Exception as e:
            print(f"Error computing embedding for text: {text[:50]}... Error: {e}")
            return [0.0] * 1536  # Fallback zero vector
    
    async def compute_dataset_embeddings(
        self, 
        dataset: List[Dict[str, Any]], 
        force_recompute: bool = False
    ) -> Dict[str, Any]:
        """
        Compute embeddings for the entire Q&A dataset asynchronously.
        
        Args:
            dataset: List of Q&A items from QADataset
            force_recompute: If True, recompute even if cache exists
            
        Returns:
            Dictionary containing embeddings and metadata
        """
        print(f"Computing embeddings for {len(dataset)} Q&A pairs asynchronously...")
        
        # Extract questions for embedding
        questions = [item['question'] for item in dataset]
        
        # Compute embeddings concurrently
        start_time = time.time()
        question_embeddings = await self.compute_batch_embeddings(questions)
        end_time = time.time()
        
        print(f"✅ Computed {len(question_embeddings)} embeddings in {end_time - start_time:.2f}s")
        
        # Create items with embeddings
        items = []
        for i, item in enumerate(dataset):
            items.append({
                'id': item['id'],
                'question': item['question'],
                'answer': item['answer'],
                'embedding': question_embeddings[i]
            })
        
        # Create metadata
        metadata = {
            'embedding_model': self.model,
            'embedding_dimension': len(question_embeddings[0]) if question_embeddings else 0,
            'total_items': len(items),
            'computation_time': end_time - start_time,
            'timestamp': time.time()
        }
        
        return {
            'items': items,
            'metadata': metadata
        }
    
    async def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings asynchronously.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Convert to numpy arrays for efficient computation
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        # Ensure result is in [0, 1] range
        return max(0.0, min(1.0, (similarity + 1) / 2))
    
    async def find_most_similar(
        self, 
        query_embedding: List[float], 
        embeddings_data: Dict[str, Any],
        top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Find the most similar items to a query embedding asynchronously.
        
        Args:
            query_embedding: Query embedding vector
            embeddings_data: Dataset with embeddings
            top_k: Number of top results to return
            
        Returns:
            List of (item, similarity_score) tuples, sorted by similarity
        """
        items = embeddings_data.get('items', [])
        
        # Compute similarities concurrently
        similarity_tasks = []
        for item in items:
            task = asyncio.create_task(
                self.compute_similarity(query_embedding, item['embedding'])
            )
            similarity_tasks.append((item, task))
        
        # Wait for all similarity computations
        similarities = []
        for item, task in similarity_tasks:
            try:
                similarity = await task
                similarities.append((item, similarity))
            except Exception as e:
                print(f"Error computing similarity: {e}")
                similarities.append((item, 0.0))
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        return similarities[:top_k]
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics asynchronously.
        
        Returns:
            Dictionary with cache statistics
        """
        async with self._cache_lock:
            return {
                'cache_size': len(self.embeddings_cache),
                'model': self.model,
                'max_concurrent': self._semaphore._value,
                'batch_size': self.batch_size
            }
    
    async def clear_cache(self) -> None:
        """Clear the embedding cache asynchronously."""
        async with self._cache_lock:
            self.embeddings_cache.clear()
    
    async def close(self) -> None:
        """Clean up resources."""
        await self.client.close()


async def test_async_embedding_service():
    """Test the async embedding service."""
    print("🧪 Testing Async Embedding Service...")
    
    try:
        service = AsyncEmbeddingService()
        
        # Test single embedding
        print("Testing single embedding...")
        embedding = await service.compute_embedding("What is DeFi?")
        print(f"✅ Single embedding computed: {len(embedding)} dimensions")
        
        # Test batch embeddings
        print("Testing batch embeddings...")
        texts = [
            "What is lending in DeFi?",
            "How does Uniswap work?",
            "What are yield farming risks?"
        ]
        embeddings = await service.compute_batch_embeddings(texts)
        print(f"✅ Batch embeddings computed: {len(embeddings)} embeddings")
        
        # Test similarity
        print("Testing similarity computation...")
        similarity = await service.compute_similarity(embeddings[0], embeddings[1])
        print(f"✅ Similarity computed: {similarity:.3f}")
        
        # Get cache stats
        stats = await service.get_cache_stats()
        print(f"✅ Cache stats: {stats}")
        
        await service.close()
        print("✅ Async embedding service test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_async_embedding_service()) 