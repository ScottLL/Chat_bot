"""
Embedding service for computing and managing text embeddings.

This module provides functionality to compute embeddings using OpenAI's
text-embedding-3-small model for semantic search in the DeFi Q&A chatbot.
"""

import os
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import openai
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
# Import configuration
from config import config

# Import services (relative imports)
from .dataset_loader import QADataset
from .cache_manager import CacheManager

# Load environment variables from .env file
load_dotenv()


class EmbeddingService:
    """Service for computing and managing text embeddings."""
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: str = None,
        batch_size: int = 100,
        max_retries: int = 3,
        timeout: int = 30,
        cache_enabled: bool = None,
        cache_dir: str = None
    ):
        """
        Initialize the embedding service.
        
        Args:
            api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
            model: Embedding model to use 
            batch_size: Number of texts to process in each batch
            max_retries: Maximum number of API retry attempts
            timeout: API request timeout in seconds
            cache_enabled: Whether to enable persistent caching of embeddings
            cache_dir: Directory for cache storage
        """
        # Use config values when not explicitly provided
        self.api_key = api_key or config.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided or set in OPENAI_API_KEY environment variable"
            )
            
        self.model = model or config.AGENT_EMBEDDING_MODEL
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout
        self.cache_enabled = cache_enabled if cache_enabled is not None else config.CACHE_ENABLED
        cache_dir = cache_dir or config.CACHE_DIR
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        # Initialize cache manager for persistent storage
        if self.cache_enabled:
            self.cache = CacheManager(cache_dir=cache_dir)
        else:
            self.cache = None
        
        # In-memory cache for session-level caching
        self.embeddings_cache: Dict[str, List[float]] = {}
        
    def compute_embedding(self, text: str) -> List[float]:
        """
        Compute embedding for a single text.
        
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
        if text_hash in self.embeddings_cache:
            return self.embeddings_cache[text_hash]
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text.strip(),
                    timeout=self.timeout
                )
                
                embedding = response.data[0].embedding
                
                # Cache the result
                self.embeddings_cache[text_hash] = embedding
                
                return embedding
                
            except openai.RateLimitError as e:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}")
                time.sleep(wait_time)
                
            except openai.APIError as e:
                print(f"OpenAI API error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(1)
                
            except Exception as e:
                print(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(1)
        
        raise Exception(f"Failed to compute embedding after {self.max_retries} attempts")
    
    def compute_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Compute embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding lists
        """
        if not texts:
            return []
            
        embeddings = []
        
        # Process in batches to respect API limits
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = []
            
            for text in batch:
                try:
                    embedding = self.compute_embedding(text)
                    batch_embeddings.append(embedding)
                    
                    # Small delay to avoid hitting rate limits
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"Failed to compute embedding for text: {text[:50]}... Error: {e}")
                    # Use zero vector as fallback
                    batch_embeddings.append([0.0] * 1536)  # text-embedding-3-small dimension
            
            embeddings.extend(batch_embeddings)
            
            # Progress indicator
            print(f"Processed {min(i + self.batch_size, len(texts))}/{len(texts)} texts")
        
        return embeddings
    
    def compute_dataset_embeddings(self, dataset: List[Dict[str, Any]], force_recompute: bool = False) -> Dict[str, Any]:
        """
        Compute embeddings for the entire Q&A dataset with persistent caching.
        
        Args:
            dataset: List of Q&A items from QADataset
            force_recompute: If True, recompute even if cache exists
            
        Returns:
            Dictionary containing embeddings and metadata
        """
        # Generate cache key based on dataset content and model
        dataset_hash = str(hash(str(sorted([(item['id'], item['question'], item['answer']) for item in dataset]))))
        cache_key = f"dataset_embeddings_{self.model}_{dataset_hash}"
        
        # Try to load from cache first
        if self.cache_enabled and not force_recompute:
            print("Checking cache for existing embeddings...")
            cached_data = self.cache.load_embeddings(cache_key)
            if cached_data:
                print(f"✅ Found cached embeddings for {len(dataset)} Q&A pairs!")
                print(f"   - Model: {cached_data['metadata'].get('embedding_model', 'unknown')}")
                print(f"   - Cached on: {time.ctime(cached_data['timestamp'])}")
                
                # Return cached embeddings in the expected format
                return {
                    'items': cached_data['embeddings'],
                    'metadata': cached_data['metadata']
                }
        
        print(f"Computing embeddings for {len(dataset)} Q&A pairs...")
        
        # Extract questions and answers
        questions = [item['question'] for item in dataset]
        answers = [item['answer'] for item in dataset]
        
        # Compute embeddings
        print("Computing question embeddings...")
        question_embeddings = self.compute_batch_embeddings(questions)
        
        print("Computing answer embeddings...")
        answer_embeddings = self.compute_batch_embeddings(answers)
        
        # Combine with original data
        items_with_embeddings = []
        for i, item in enumerate(dataset):
            items_with_embeddings.append({
                'id': item['id'],
                'question': item['question'],
                'answer': item['answer'],
                'question_embedding': question_embeddings[i],
                'answer_embedding': answer_embeddings[i]
            })
        
        # Prepare results
        results = {
            'items': items_with_embeddings,
            'metadata': {
                'total_items': len(dataset),
                'embedding_model': self.model,
                'embedding_dimension': len(question_embeddings[0]) if question_embeddings else 0,
                'computed_at': time.time(),
                'dataset_hash': dataset_hash
            }
        }
        
        # Save to persistent cache
        if self.cache_enabled:
            cache_metadata = {
                'embedding_model': self.model,
                'total_items': len(dataset),
                'embedding_dimension': len(question_embeddings[0]) if question_embeddings else 0,
                'dataset_hash': dataset_hash
            }
            
            if self.cache.save_embeddings(cache_key, items_with_embeddings, cache_metadata):
                print(f"💾 Successfully cached embeddings for future use!")
            else:
                print(f"⚠️ Failed to save embeddings to cache")
        
        print(f"Successfully computed embeddings for all {len(dataset)} items!")
        return results
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (-1 to 1)
        """
        if len(embedding1) != len(embedding2):
            raise ValueError("Embeddings must have the same dimension")
            
        # Convert to numpy arrays for efficient computation
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def find_most_similar(
        self, 
        query_embedding: List[float], 
        embeddings_data: Dict[str, Any],
        top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Find most similar items to a query embedding.
        
        Args:
            query_embedding: Query embedding vector
            embeddings_data: Dataset with embeddings from compute_dataset_embeddings
            top_k: Number of top results to return
            
        Returns:
            List of (item, similarity_score) tuples sorted by similarity
        """
        similarities = []
        
        for item in embeddings_data['items']:
            # Compare with question embedding
            question_sim = self.compute_similarity(
                query_embedding, 
                item['question_embedding']
            )
            
            similarities.append((item, question_sim))
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the embedding cache."""
        return {
            'cache_size': len(self.embeddings_cache),
            'model': self.model,
            'batch_size': self.batch_size
        }


def main():
    """Test the embedding service."""
    # Note: You need to set OPENAI_API_KEY environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-api-key-here'")
        return
    
    try:
        # Initialize services
        dataset_loader = QADataset()
        embedding_service = EmbeddingService(api_key=api_key)
        
        # Load dataset
        print("Loading dataset...")
        dataset = dataset_loader.load_dataset()
        
        # Test single embedding
        print("\nTesting single embedding...")
        test_text = "What is DeFi?"
        embedding = embedding_service.compute_embedding(test_text)
        print(f"Embedding dimension: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
        
        # Uncomment to test full dataset (requires API key and may incur costs)
        # print("\nComputing embeddings for entire dataset...")
        # embeddings_data = embedding_service.compute_dataset_embeddings(dataset[:3])  # Test with first 3 items
        # print(f"Computed embeddings for {len(embeddings_data['items'])} items")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 