"""
Dataset loader for DeFi Q&A data.

This module provides functionality to load and validate the Q&A dataset
used for semantic search in the DeFi chatbot application.
"""

import json
import os
from typing import List, Dict, Any
from pathlib import Path


class QADataset:
    """Class to handle loading and managing Q&A dataset."""
    
    def __init__(self, data_path: str = "../../data/defi_qa_dataset.json"):
        """
        Initialize the dataset loader.
        
        Args:
            data_path: Path to the JSON dataset file
        """
        self.data_path = data_path
        self.data: List[Dict[str, Any]] = []
        
    def load_dataset(self) -> List[Dict[str, Any]]:
        """
        Load the Q&A dataset from JSON file.
        
        Returns:
            List of Q&A pairs with id, question, and answer fields
            
        Raises:
            FileNotFoundError: If dataset file doesn't exist
            json.JSONDecodeError: If JSON is malformed
            ValueError: If dataset structure is invalid
        """
        # Handle multiple potential paths for Docker and local environments
        possible_paths = [
            Path(__file__).parent / self.data_path,  # Original relative path
            Path("/app/data/defi_qa_dataset.json"),  # Docker absolute path
            Path("data/defi_qa_dataset.json"),       # Simple relative path
            Path("./data/defi_qa_dataset.json"),     # Current dir relative
        ]
        
        dataset_path = None
        for path in possible_paths:
            if path.exists():
                dataset_path = path
                break
        
        if dataset_path is None:
            searched_paths = '\n'.join(str(p) for p in possible_paths)
            raise FileNotFoundError(f"Dataset file not found. Searched paths:\n{searched_paths}")
            
        try:
            with open(dataset_path, 'r', encoding='utf-8') as file:
                self.data = json.load(file)
                
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in dataset file: {e}")
            
        # Validate dataset structure
        self._validate_dataset()
        
        print(f"Successfully loaded {len(self.data)} Q&A pairs")
        return self.data
    
    def _validate_dataset(self) -> None:
        """
        Validate that dataset has required structure.
        
        Raises:
            ValueError: If dataset structure is invalid
        """
        if not isinstance(self.data, list):
            raise ValueError("Dataset must be a list of Q&A pairs")
            
        if len(self.data) == 0:
            raise ValueError("Dataset is empty")
            
        required_fields = {'id', 'question', 'answer'}
        
        for idx, item in enumerate(self.data):
            if not isinstance(item, dict):
                raise ValueError(f"Item {idx} is not a dictionary")
                
            missing_fields = required_fields - set(item.keys())
            if missing_fields:
                raise ValueError(
                    f"Item {idx} missing required fields: {missing_fields}"
                )
                
            # Check for empty values
            for field in required_fields:
                if not item[field] or not isinstance(item[field], str):
                    raise ValueError(
                        f"Item {idx} has invalid {field}: must be non-empty string"
                    )
    
    def get_questions(self) -> List[str]:
        """Get all questions from the dataset."""
        return [item['question'] for item in self.data]
    
    def get_answers(self) -> List[str]:
        """Get all answers from the dataset."""
        return [item['answer'] for item in self.data]
    
    def get_qa_pairs(self) -> List[Dict[str, str]]:
        """Get question-answer pairs."""
        return [
            {
                'question': item['question'], 
                'answer': item['answer']
            } 
            for item in self.data
        ]
    
    def get_item_by_id(self, item_id: str) -> Dict[str, Any]:
        """
        Get Q&A item by ID.
        
        Args:
            item_id: ID of the item to retrieve
            
        Returns:
            Q&A item dictionary
            
        Raises:
            ValueError: If ID not found
        """
        for item in self.data:
            if item['id'] == item_id:
                return item
                
        raise ValueError(f"Item with ID '{item_id}' not found")
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get statistics about the dataset."""
        if not self.data:
            return {"total_items": 0}
            
        questions = self.get_questions()
        answers = self.get_answers()
        
        return {
            "total_items": len(self.data),
            "avg_question_length": sum(len(q) for q in questions) / len(questions),
            "avg_answer_length": sum(len(a) for a in answers) / len(answers),
            "question_length_range": (
                min(len(q) for q in questions),
                max(len(q) for q in questions)
            ),
            "answer_length_range": (
                min(len(a) for a in answers),
                max(len(a) for a in answers)
            )
        }


def main():
    """Test the dataset loader."""
    try:
        loader = QADataset()
        dataset = loader.load_dataset()
        
        print(f"Dataset loaded successfully!")
        print(f"Total items: {len(dataset)}")
        
        # Print first item as example
        if dataset:
            print(f"\nFirst item:")
            print(f"ID: {dataset[0]['id']}")
            print(f"Question: {dataset[0]['question']}")
            print(f"Answer: {dataset[0]['answer'][:100]}...")
            
        # Print statistics
        stats = loader.get_dataset_stats()
        print(f"\nDataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")


if __name__ == "__main__":
    main() 