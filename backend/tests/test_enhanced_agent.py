"""
Tests for the enhanced async DeFi Q&A agent with conversation support.

This module tests the agent's ability to:
1. Classify questions as DeFi-related or general conversation
2. Handle general conversation appropriately
3. Add reference labels to dataset responses
4. Route questions correctly through the workflow
"""

import pytest
import pytest_asyncio
import asyncio
import sys
import os
from unittest.mock import Mock, AsyncMock, patch

# Add parent directory to path to enable imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents.async_defi_qa_agent import (
    AsyncDeFiQAAgent, AsyncAgentState, AsyncAgentFactory, async_agent_factory
)
from services.session_manager import SessionContext


@pytest_asyncio.fixture
async def mock_session():
    """Create a mock session context for testing."""
    return SessionContext(session_id="test-session", user_id="test-user")


@pytest_asyncio.fixture
async def mock_embedding_service():
    """Create a mock embedding service."""
    service = Mock()
    service.compute_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
    service.find_most_similar = AsyncMock(return_value=[
        ({'id': 'q1', 'question': 'What is DeFi?', 'answer': 'DeFi is decentralized finance'}, 0.85),
        ({'id': 'q2', 'question': 'How does staking work?', 'answer': 'Staking involves locking tokens'}, 0.75)
    ])
    service.model = "text-embedding-ada-002"
    return service


@pytest_asyncio.fixture
async def mock_dataset():
    """Create a mock dataset."""
    return [
        {'id': 'q1', 'question': 'What is DeFi?', 'answer': 'DeFi is decentralized finance'},
        {'id': 'q2', 'question': 'How does staking work?', 'answer': 'Staking involves locking tokens'}
    ]


@pytest_asyncio.fixture
async def mock_embedding_data():
    """Create mock embedding data."""
    return {
        'items': [
            {'id': 'q1', 'question': 'What is DeFi?', 'answer': 'DeFi is decentralized finance', 'question_embedding': [0.1, 0.2, 0.3]},
            {'id': 'q2', 'question': 'How does staking work?', 'answer': 'Staking involves locking tokens', 'question_embedding': [0.2, 0.3, 0.4]}
        ],
        'model': 'text-embedding-ada-002',
        'dimension': 3
    }


@pytest_asyncio.fixture
async def test_agent(mock_session, mock_dataset, mock_embedding_data, mock_embedding_service):
    """Create a test agent instance."""
    with patch('agents.async_defi_qa_agent.ChatOpenAI') as mock_llm_class:
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(return_value=Mock(content="Hello! I'm here to help with DeFi questions."))
        mock_llm_class.return_value = mock_llm
        
        agent = AsyncDeFiQAAgent(
            session_context=mock_session,
            dataset=mock_dataset,
            embedding_data=mock_embedding_data,
            embedding_service=mock_embedding_service
        )
        return agent


class TestQuestionClassification:
    """Test question classification functionality."""
    
    @pytest.mark.asyncio
    async def test_classify_greeting(self, test_agent):
        """Test classification of greeting messages."""
        state = AsyncAgentState(
            session_context=test_agent.session_context,
            question="Hello, how are you?",
            processed_question="Hello, how are you?"
        )
        
        result = await test_agent._classify_question_node(state)
        assert result.question_type == "general"
        assert result.error_message == ""
    
    @pytest.mark.asyncio
    async def test_classify_defi_question(self, test_agent):
        """Test classification of DeFi-related questions."""
        state = AsyncAgentState(
            session_context=test_agent.session_context,
            question="What is yield farming in DeFi?",
            processed_question="What is yield farming in DeFi?"
        )
        
        result = await test_agent._classify_question_node(state)
        assert result.question_type == "defi"
        assert result.error_message == ""
    
    @pytest.mark.asyncio
    async def test_classify_thank_you(self, test_agent):
        """Test classification of thank you messages."""
        state = AsyncAgentState(
            session_context=test_agent.session_context,
            question="Thanks for your help!",
            processed_question="Thanks for your help!"
        )
        
        result = await test_agent._classify_question_node(state)
        assert result.question_type == "general"
        assert result.error_message == ""
    
    @pytest.mark.asyncio
    async def test_classify_ambiguous_defaults_to_defi(self, test_agent):
        """Test that ambiguous questions default to DeFi."""
        state = AsyncAgentState(
            session_context=test_agent.session_context,
            question="How does this work?",
            processed_question="How does this work?"
        )
        
        result = await test_agent._classify_question_node(state)
        assert result.question_type == "defi"
        assert result.error_message == ""


class TestGeneralConversation:
    """Test general conversation handling."""
    
    @pytest.mark.asyncio
    async def test_general_conversation_node(self, test_agent):
        """Test general conversation response generation."""
        state = AsyncAgentState(
            session_context=test_agent.session_context,
            question="Hello!",
            processed_question="Hello!",
            question_type="general"
        )
        
        result = await test_agent._general_conversation_node(state)
        
        assert result.error_message == ""
        assert result.response == "Hello! I'm here to help with DeFi questions."
        assert result.is_dataset_response == False
        assert result.metadata['response_type'] == 'general_conversation'
        assert result.metadata['confidence_score'] == 1.0
        assert result.processing_stage == "completed"


class TestDeFiQuestionHandling:
    """Test DeFi question handling with reference labeling."""
    
    @pytest.mark.asyncio
    async def test_semantic_search_node(self, test_agent):
        """Test semantic search functionality."""
        state = AsyncAgentState(
            session_context=test_agent.session_context,
            question="What is DeFi?",
            processed_question="What is DeFi?",
            embeddings_data=test_agent.embedding_data
        )
        
        result = await test_agent._semantic_search_node(state)
        
        assert result.error_message == ""
        assert len(result.similar_items) == 2
        assert len(result.similarity_scores) == 2
        assert result.processing_stage == "semantic_search"
    
    @pytest.mark.asyncio
    async def test_select_answer_with_metadata(self, test_agent):
        """Test answer selection with proper metadata generation."""
        state = AsyncAgentState(
            session_context=test_agent.session_context,
            question="What is DeFi?",
            processed_question="What is DeFi?",
            similar_items=[
                {'id': 'q1', 'question': 'What is DeFi?', 'answer': 'DeFi is decentralized finance'}
            ],
            similarity_scores=[0.85]
        )
        
        result = await test_agent._select_answer_node(state)
        
        assert result.error_message == ""
        assert result.response == "DeFi is decentralized finance"
        assert result.is_dataset_response == True
        assert result.metadata['response_type'] == 'dataset_answer'
        assert result.metadata['source_id'] == 'q1'
        assert result.metadata['confidence_score'] == 0.85
        assert 'dataset_reference' in result.metadata
    
    @pytest.mark.asyncio
    async def test_generate_response_with_reference(self, test_agent):
        """Test response generation with reference labeling."""
        state = AsyncAgentState(
            session_context=test_agent.session_context,
            question="What is DeFi?",
            response="DeFi is decentralized finance",
            is_dataset_response=True,
            metadata={
                'dataset_reference': {'id': 'q1', 'original_question': 'What is DeFi?'},
                'confidence_score': 0.85
            }
        )
        
        result = await test_agent._generate_response_node(state)
        
        assert result.error_message == ""
        assert "📚 **Reference**: Dataset entry #q1 (Confidence: 0.85)" in result.response
        assert result.processing_stage == "completed"


class TestWorkflowRouting:
    """Test workflow routing based on question type."""
    
    @pytest.mark.asyncio
    async def test_route_general_question(self, test_agent):
        """Test routing of general questions to conversation handler."""
        result = test_agent._route_by_question_type(AsyncAgentState(
            session_context=test_agent.session_context,
            question="Hello",
            question_type="general"
        ))
        assert result == "general"
    
    @pytest.mark.asyncio
    async def test_route_defi_question(self, test_agent):
        """Test routing of DeFi questions to semantic search."""
        result = test_agent._route_by_question_type(AsyncAgentState(
            session_context=test_agent.session_context,
            question="What is staking?",
            question_type="defi"
        ))
        assert result == "defi"
    
    @pytest.mark.asyncio
    async def test_route_error_state(self, test_agent):
        """Test routing when there's an error."""
        result = test_agent._route_by_question_type(AsyncAgentState(
            session_context=test_agent.session_context,
            question="Test",
            question_type="general",
            error_message="Some error"
        ))
        assert result == "error"


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_general_conversation_workflow(self, test_agent):
        """Test complete workflow for general conversation."""
        result = await test_agent.ask_question_async("Hello, how are you?")
        
        assert result['error'] is None
        assert result['question_type'] == "general"
        assert result['is_dataset_response'] == False
        assert 'help with DeFi questions' in result['response']
        assert result['processing_time'] > 0
    
    @pytest.mark.asyncio
    async def test_complete_defi_question_workflow(self, test_agent):
        """Test complete workflow for DeFi questions."""
        result = await test_agent.ask_question_async("What is yield farming in DeFi?")
        
        assert result['error'] is None
        assert result['question_type'] == "defi"
        assert result['is_dataset_response'] == True
        assert "📚 **Reference**" in result['response']
        assert result['processing_time'] > 0
    
    @pytest.mark.asyncio
    async def test_streaming_with_metadata(self, test_agent):
        """Test streaming response with proper metadata."""
        chunks = []
        async for chunk in test_agent.ask_question_stream_async("Hello!"):
            chunks.append(chunk)
        
        # Check metadata chunk
        metadata_chunk = next((c for c in chunks if c['type'] == 'metadata'), None)
        assert metadata_chunk is not None
        assert 'question_type' in metadata_chunk
        assert 'is_dataset_response' in metadata_chunk
        
        # Check completion chunk
        complete_chunk = next((c for c in chunks if c['type'] == 'complete'), None)
        assert complete_chunk is not None
        assert 'question_type' in complete_chunk
        assert 'is_dataset_response' in complete_chunk


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_empty_question_error(self, test_agent):
        """Test handling of empty questions."""
        result = await test_agent.ask_question_async("")
        
        assert result['error'] is None  # Should be handled gracefully
        assert "Please provide a question or message" in result['response']
    
    @pytest.mark.asyncio
    async def test_very_short_question_error(self, test_agent):
        """Test handling of very short questions."""
        result = await test_agent.ask_question_async("a")
        
        assert result['error'] is None  # Should be handled gracefully
        assert "too short" in result['response']
    
    @pytest.mark.asyncio
    async def test_no_defi_results_found(self, test_agent):
        """Test handling when no DeFi results are found."""
        # Mock the embedding service to return no results
        test_agent.embedding_service.find_most_similar = AsyncMock(return_value=[])
        
        result = await test_agent.ask_question_async("What is some obscure DeFi protocol?")
        
        assert result['error'] is None  # Should be handled gracefully
        assert "couldn't find a relevant answer" in result['response']


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 