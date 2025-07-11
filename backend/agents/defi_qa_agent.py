"""
DeFi Q&A LangGraph Agent

This module implements a LangGraph-based agent for answering DeFi-related questions
using semantic search over a pre-loaded dataset. The agent orchestrates the workflow
through nodes and transitions and supports both DeFi questions and general conversation.
"""

import os
import time
import asyncio
from typing import TypedDict, Annotated, Sequence, Dict, Any, List, Optional
from operator import add as add_messages

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

# Import services
from services.dataset_loader import QADataset
from services.embedding_service import EmbeddingService
from services.cache_manager import CacheManager

# Import configuration
from config import config

# Load environment variables
load_dotenv()


class AgentState(TypedDict):
    """
    State definition for the DeFi Q&A Agent.
    
    This state tracks the conversation flow and data needed
    for semantic search and response generation.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_question: str
    parsed_question: str
    question_type: str  # 'defi' or 'general'
    embedding_results: Optional[Dict[str, Any]]
    retrieved_qa_pairs: List[Dict[str, Any]]
    selected_answer: str
    similarity_scores: List[float]
    error_message: Optional[str]
    processing_stage: str
    is_dataset_response: bool  # Track if response came from dataset
    metadata: Dict[str, Any]


class DeFiQAAgent:
    """
    Main LangGraph agent for DeFi Q&A functionality with conversation support.
    
    Orchestrates the complete workflow from input parsing through
    semantic retrieval to streaming responses, and handles both
    DeFi questions and general conversation.
    """
    
    def __init__(
        self,
        llm_model: str = None,
        embedding_model: str = None,
        cache_enabled: bool = None,
        similarity_threshold: float = None,
        max_results: int = None
    ):
        """
        Initialize the DeFi Q&A Agent.
        
        Args:
            llm_model: OpenAI model for response generation
            embedding_model: OpenAI model for embeddings
            cache_enabled: Whether to enable embedding caching
            similarity_threshold: Minimum similarity score for results
            max_results: Maximum number of results to return
        """
        # Use config values if not explicitly provided
        self.llm_model = llm_model or config.AGENT_LLM_MODEL
        self.embedding_model = embedding_model or config.AGENT_EMBEDDING_MODEL
        self.similarity_threshold = similarity_threshold if similarity_threshold is not None else config.AGENT_SIMILARITY_THRESHOLD
        self.max_results = max_results if max_results is not None else config.AGENT_MAX_RESULTS
        cache_enabled = cache_enabled if cache_enabled is not None else config.AGENT_CACHE_ENABLED
        
        # Initialize components
        self.dataset_loader = QADataset()
        self.embedding_service = EmbeddingService(
            model=self.embedding_model,
            cache_enabled=cache_enabled
        )
        self.cache_manager = CacheManager(cache_dir=config.CACHE_DIR) if cache_enabled else None
        
        # Initialize LLM for DeFi responses
        self.llm = ChatOpenAI(
            model=self.llm_model,
            temperature=0,  # Minimize hallucination for factual Q&A
            streaming=True
        )
        
        # Initialize LLM for general conversation
        self.conversation_llm = ChatOpenAI(
            model=self.llm_model,
            temperature=0.7,  # Higher temperature for more natural conversation
            streaming=False
        )
        
        # Load and prepare dataset
        self._load_dataset()
        
        # Build the state graph
        self.graph = self._build_graph()
        self.agent = self.graph.compile()
    
    def _load_dataset(self):
        """Load and prepare the Q&A dataset with embeddings."""
        print("🔄 Loading DeFi Q&A dataset...")
        
        try:
            # Load dataset
            self.dataset = self.dataset_loader.load_dataset()
            print(f"✅ Loaded {len(self.dataset)} Q&A pairs")
            
            # Compute/load embeddings
            print("🔄 Computing embeddings...")
            self.embedding_data = self.embedding_service.compute_dataset_embeddings(
                self.dataset
            )
            print(f"✅ Embeddings ready: {self.embedding_data['metadata']['embedding_dimension']}D")
            
        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            raise
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph with all nodes and transitions."""
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("parse_input", self.parse_input_node)
        graph.add_node("classify_question", self.classify_question_node)
        graph.add_node("semantic_search", self.semantic_search_node)
        graph.add_node("select_answer", self.select_answer_node)
        graph.add_node("generate_response", self.generate_response_node)
        graph.add_node("general_conversation", self.general_conversation_node)
        graph.add_node("handle_error", self.handle_error_node)
        
        # Define edges and transitions
        graph.set_entry_point("parse_input")
        
        # From parse_input, either classify question or handle error
        graph.add_conditional_edges(
            "parse_input",
            self._should_continue_from_parsing,
            {
                "continue": "classify_question",
                "error": "handle_error"
            }
        )
        
        # From classify_question, route based on question type
        graph.add_conditional_edges(
            "classify_question",
            self._route_by_question_type,
            {
                "defi": "semantic_search",
                "general": "general_conversation",
                "error": "handle_error"
            }
        )
        
        # From semantic_search, either select answer or handle no results
        graph.add_conditional_edges(
            "semantic_search",
            self._should_continue_from_search,
            {
                "found_results": "select_answer",
                "no_results": "handle_error"
            }
        )
        
        # From select_answer, go to generate_response
        graph.add_edge("select_answer", "generate_response")
        
        # All final nodes end the flow
        graph.add_edge("generate_response", END)
        graph.add_edge("general_conversation", END)
        graph.add_edge("handle_error", END)
        
        return graph
    
    # =============================================================================
    # NODE IMPLEMENTATIONS
    # =============================================================================
    
    def parse_input_node(self, state: AgentState) -> AgentState:
        """
        Parse and validate user input.
        
        This node handles:
        - Extracting the user question from messages
        - Basic input validation  
        - Question preprocessing and cleanup
        """
        try:
            # Extract user question from the latest human message
            user_message = None
            for message in reversed(state['messages']):
                if isinstance(message, HumanMessage):
                    user_message = message.content
                    break
            
            if not user_message:
                return {
                    **state,
                    'error_message': "No user question found in messages",
                    'processing_stage': "input_parsing_failed"
                }
            
            # Basic validation
            if not user_message.strip():
                return {
                    **state,
                    'error_message': "Question cannot be empty",
                    'processing_stage': "input_parsing_failed"
                }
            
            if len(user_message.strip()) < 2:  # Reduced minimum length for greetings
                return {
                    **state,
                    'error_message': "Question is too short. Please provide more details.",
                    'processing_stage': "input_parsing_failed"
                }
            
            # Clean and preprocess the question
            parsed_question = self._preprocess_question(user_message.strip())
            
            return {
                **state,
                'user_question': user_message.strip(),
                'parsed_question': parsed_question,
                'processing_stage': "input_parsed",
                'error_message': None,
                'is_dataset_response': False,
                'metadata': {}
            }
            
        except Exception as e:
            return {
                **state,
                'error_message': f"Input parsing error: {str(e)}",
                'processing_stage': "input_parsing_failed"
            }
    
    def classify_question_node(self, state: AgentState) -> AgentState:
        """
        Classify whether the question is DeFi-related or general conversation.
        
        This node handles:
        - Analyzing question content for DeFi keywords
        - Detecting general conversation patterns
        - Setting the question type for routing
        """
        try:
            question = state['parsed_question'].lower()
            
            # Simple classification logic
            defi_keywords = [
                'defi', 'decentralized finance', 'lending', 'borrowing', 'staking', 'yield farming',
                'liquidity', 'pool', 'dex', 'exchange', 'swap', 'token', 'ethereum', 'bitcoin',
                'blockchain', 'smart contract', 'protocol', 'aave', 'compound', 'uniswap',
                'pancakeswap', 'sushiswap', 'maker', 'dao', 'governance', 'farming', 'mining',
                'nft', 'vault', 'apr', 'apy', 'slippage', 'impermanent loss', 'flashloan',
                'collateral', 'liquidation', 'oracle', 'amm', 'automated market maker'
            ]
            
            general_patterns = [
                'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
                'how are you', 'what\'s up', 'how do you do', 'nice to meet you',
                'thank you', 'thanks', 'bye', 'goodbye', 'see you', 'weather',
                'how old are you', 'what\'s your name', 'who are you', 'what can you do'
            ]
            
            # Check for DeFi keywords
            contains_defi_keywords = any(keyword in question for keyword in defi_keywords)
            
            # Check for general conversation patterns
            contains_general_patterns = any(pattern in question for pattern in general_patterns)
            
            # Classification logic
            if contains_general_patterns and not contains_defi_keywords:
                question_type = "general"
            elif contains_defi_keywords:
                question_type = "defi"
            elif len(question.split()) <= 3 and any(word in question for word in ['hi', 'hello', 'hey', 'thanks', 'bye']):
                question_type = "general"
            else:
                # Default to DeFi for ambiguous cases (existing behavior)
                question_type = "defi"
            
            return {
                **state,
                'question_type': question_type,
                'processing_stage': "question_classified"
            }
            
        except Exception as e:
            return {
                **state,
                'error_message': f"Error classifying question: {str(e)}",
                'processing_stage': "classification_failed"
            }
    
    def general_conversation_node(self, state: AgentState) -> AgentState:
        """
        Handle general conversation using LLM.
        
        This node handles:
        - Generating natural conversation responses
        - Maintaining friendly and helpful tone
        - Suggesting DeFi topics when appropriate
        """
        try:
            # Create conversation prompt
            conversation_prompt = f"""You are a friendly and helpful AI assistant specialized in DeFi (Decentralized Finance). 
You can have general conversations but always maintain a professional and helpful tone.

User message: {state['parsed_question']}

Please respond naturally to this message. If it's a greeting, be warm and welcoming. 
If it's a general question, provide a helpful response while mentioning that you specialize in DeFi topics.
Keep your response concise and friendly.

If the user seems to be asking about DeFi topics, gently suggest they can ask specific DeFi questions for detailed answers."""
            
            # Get response from LLM
            response = self.conversation_llm.invoke(conversation_prompt)
            response_content = response.content
            
            # Create response message
            response_message = AIMessage(content=response_content)
            
            return {
                **state,
                'messages': [response_message],
                'processing_stage': "general_conversation_completed",
                'is_dataset_response': False,
                'metadata': {
                    'response_type': 'general_conversation',
                    'llm_model': self.llm_model,
                    'confidence_score': 1.0
                }
            }
            
        except Exception as e:
            return {
                **state,
                'error_message': f"Error in general conversation: {str(e)}",
                'processing_stage': "general_conversation_failed"
            }
    
    def semantic_search_node(self, state: AgentState) -> AgentState:
        """
        Perform semantic search over the Q&A dataset.
        
        This node handles:
        - Computing embedding for the user question
        - Finding similar Q&A pairs using cosine similarity
        - Filtering results by similarity threshold
        """
        try:
            question = state['parsed_question']
            
            # Compute embedding for the user question
            question_embedding = self.embedding_service.compute_embedding(question)
            
            # Find similar Q&A pairs
            similarities = []
            for item in self.embedding_data['items']:
                # Calculate similarity with question embeddings
                similarity = self.embedding_service.compute_similarity(
                    question_embedding,
                    item['question_embedding']
                )
                
                if similarity >= self.similarity_threshold * 0.4:  # Lowered threshold slightly
                    similarities.append((item, similarity))
            
            # Sort by similarity score (highest first)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Take top results
            top_results = similarities[:self.max_results]
            retrieved_qa_pairs = [item for item, score in top_results]
            similarity_scores = [score for item, score in top_results]
            
            return {
                **state,
                'retrieved_qa_pairs': retrieved_qa_pairs,
                'similarity_scores': similarity_scores,
                'processing_stage': "semantic_search_completed" if top_results else "no_results_found"
            }
            
        except Exception as e:
            return {
                **state,
                'error_message': f"Semantic search error: {str(e)}",
                'processing_stage': "semantic_search_failed"
            }
    
    def select_answer_node(self, state: AgentState) -> AgentState:
        """
        Select the best answer from retrieved results, potentially including multiple references.
        
        This node handles:
        - Evaluating retrieved Q&A pairs
        - Selecting the most relevant answer
        - Preparing context for response generation with metadata including multiple references
        """
        try:
            retrieved_pairs = state['retrieved_qa_pairs']
            similarity_scores = state['similarity_scores']
            
            if not retrieved_pairs:
                return {
                    **state,
                    'error_message': "No relevant answers found",
                    'processing_stage': "answer_selection_failed"
                }
            
            # Select the highest similarity answer as primary
            best_qa_pair = retrieved_pairs[0]
            best_score = similarity_scores[0]
            
            # Collect multiple references if they're high quality
            references = []
            reference_threshold = best_score * 0.85  # Include references within 15% of best score
            
            for i, (qa_pair, score) in enumerate(zip(retrieved_pairs, similarity_scores)):
                if score >= reference_threshold and len(references) < 3:  # Max 3 references
                    references.append({
                        'id': qa_pair['id'],
                        'original_question': qa_pair['question'],
                        'confidence': score,
                        'is_primary': (i == 0)
                    })
            
            return {
                **state,
                'selected_answer': best_qa_pair['answer'],
                'processing_stage': "answer_selected",
                'is_dataset_response': True,
                'metadata': {
                    'response_type': 'dataset_answer',
                    'source_id': best_qa_pair['id'],
                    'source_question': best_qa_pair['question'],
                    'confidence_score': best_score,
                    'total_candidates': len(retrieved_pairs),
                    'all_scores': similarity_scores[:3],
                    'dataset_references': references,  # Multiple references
                    'dataset_reference': {  # Keep single reference for compatibility
                        'id': best_qa_pair['id'],
                        'original_question': best_qa_pair['question']
                    }
                }
            }
            
        except Exception as e:
            return {
                **state,
                'error_message': f"Answer selection error: {str(e)}",
                'processing_stage': "answer_selection_failed"
            }
    
    def generate_response_node(self, state: AgentState) -> AgentState:
        """
        Generate the final response with proper reference labeling.
        
        This node handles:
        - Formatting the response with reference information
        - Adding confidence scores
        - Preparing for streaming output
        """
        try:
            user_question = state['user_question']
            selected_answer = state['selected_answer']
            metadata = state.get('metadata', {})
            
            # For dataset responses, format without inline confidence
            response_content = selected_answer
            if state.get('is_dataset_response') and metadata:
                references = metadata.get('dataset_references', [])
                
                if len(references) > 1:
                    # Multiple references
                    reference_lines = []
                    for ref in references:
                        reference_lines.append(f"📚 **Reference**: Dataset entry #{ref['id']}")
                    
                    # Add all references
                    response_content = f"{selected_answer}\n\n" + "\n".join(reference_lines)
                    
                    # Keep detailed reference info in metadata for frontend processing
                    metadata['display_references'] = {
                        'items': [{'text': f"Dataset entry #{ref['id']}", 'source_id': ref['id']} for ref in references],
                        'should_show': True,
                        'count': len(references)
                    }
                elif len(references) == 1:
                    # Single reference (existing behavior)
                    source_id = references[0]['id']
                    
                    # Add reference text without confidence (confidence shown in UI bar)
                    response_content = f"{selected_answer}\n\n📚 **Reference**: Dataset entry #{source_id}"
                    
                    # Keep detailed reference info in metadata for frontend processing
                    metadata['display_references'] = {
                        'items': [{'text': f"Dataset entry #{source_id}", 'source_id': source_id}],
                        'should_show': True,
                        'count': 1
                    }
                else:
                    # Fallback to old metadata structure
                    reference_info = metadata.get('dataset_reference', {})
                    source_id = reference_info.get('id', 'Unknown')
                    
                    response_content = f"{selected_answer}\n\n📚 **Reference**: Dataset entry #{source_id}"
                    
                    metadata['display_references'] = {
                        'items': [{'text': f"Dataset entry #{source_id}", 'source_id': source_id}],
                        'should_show': True,
                        'count': 1
                    }
            else:
                # For general conversation, no reference needed
                if not metadata:
                    metadata = {}
                metadata['display_references'] = {
                    'should_show': False,
                    'count': 0
                }
            
            response_message = AIMessage(content=response_content)
            
            return {
                **state,
                'messages': [response_message],
                'metadata': metadata,
                'processing_stage': "response_generated"
            }
            
        except Exception as e:
            return {
                **state,
                'error_message': f"Response generation error: {str(e)}",
                'processing_stage': "response_generation_failed"
            }
    
    def handle_error_node(self, state: AgentState) -> AgentState:
        """
        Handle errors and provide helpful feedback.
        
        This node handles:
        - Error message formatting
        - Providing suggestions for better queries
        - Graceful failure responses
        """
        try:
            error_message = state.get('error_message', 'Unknown error occurred')
            processing_stage = state.get('processing_stage', 'unknown_stage')
            
            # Provide context-specific error messages
            if processing_stage == "input_parsing_failed":
                response = f"I couldn't understand your message. {error_message}"
            elif processing_stage == "no_results_found":
                response = ("I couldn't find any relevant information about your DeFi question. "
                           "Try rephrasing your question or asking about DeFi topics like "
                           "lending, staking, DEXs, or yield farming.")
            else:
                response = f"I encountered an error while processing your message: {error_message}"
            
            error_response = AIMessage(content=response)
            
            return {
                **state,
                'messages': [error_response],
                'processing_stage': "error_handled"
            }
            
        except Exception as e:
            # Fallback error handling
            fallback_response = AIMessage(
                content="I'm sorry, I encountered an unexpected error. Please try again."
            )
            return {
                **state,
                'messages': [fallback_response],
                'processing_stage': "error_handled"
            }
    
    # =============================================================================
    # HELPER METHODS
    # =============================================================================
    
    def _preprocess_question(self, question: str) -> str:
        """Preprocess and clean the user question."""
        # Basic preprocessing
        cleaned = question.strip()
        
        # Remove extra whitespace
        cleaned = ' '.join(cleaned.split())
        
        # Ensure question ends with appropriate punctuation
        if not cleaned.endswith(('?', '.', '!')):
            cleaned += '?'
        
        return cleaned
    
    def _should_continue_from_parsing(self, state: AgentState) -> str:
        """Determine next step after input parsing."""
        if state.get('error_message'):
            return "error"
        return "continue"
    
    def _route_by_question_type(self, state: AgentState) -> str:
        """Route based on question classification."""
        if state.get('error_message'):
            return "error"
        return state.get('question_type', 'defi')
    
    def _should_continue_from_search(self, state: AgentState) -> str:
        """Determine next step after semantic search."""
        if state.get('retrieved_qa_pairs'):
            return "found_results"
        return "no_results"
    
    # =============================================================================
    # PUBLIC INTERFACE
    # =============================================================================
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get a response from the agent.
        
        Args:
            question: User's question (can be DeFi-related or general conversation)
            
        Returns:
            Dict containing the response and metadata
        """
        try:
            # Create initial state
            initial_state = {
                'messages': [HumanMessage(content=question)],
                'user_question': '',
                'parsed_question': '',
                'question_type': '',
                'embedding_results': None,
                'retrieved_qa_pairs': [],
                'selected_answer': '',
                'similarity_scores': [],
                'error_message': None,
                'processing_stage': 'initialized',
                'is_dataset_response': False,
                'metadata': {}
            }
            
            # Run the agent
            result = self.agent.invoke(initial_state)
            
            # Extract response
            response_message = result['messages'][-1] if result['messages'] else None
            response_content = response_message.content if response_message else "No response generated"
            
            return {
                'response': response_content,
                'processing_stage': result.get('processing_stage', 'unknown'),
                'similarity_scores': result.get('similarity_scores', []),
                'question_type': result.get('question_type', 'unknown'),
                'is_dataset_response': result.get('is_dataset_response', False),
                'metadata': result.get('metadata', {}),
                'error': result.get('error_message')
            }
            
        except Exception as e:
            return {
                'response': f"I'm sorry, I encountered an error: {str(e)}",
                'processing_stage': 'error',
                'similarity_scores': [],
                'question_type': 'unknown',
                'is_dataset_response': False,
                'metadata': {},
                'error': str(e)
            }


def main():
    """Test the enhanced DeFi Q&A Agent."""
    print("🚀 Initializing Enhanced DeFi Q&A Agent...")
    
    try:
        agent = DeFiQAAgent()
        print("✅ Agent initialized successfully!")
        
        # Test questions - mix of general and DeFi
        test_questions = [
            "Hello, how are you?",
            "What is the largest lending pool on Aave?",
            "Thanks for your help!",
            "How does Uniswap V3 work?",
            "Good morning!",
            "What are the risks of yield farming?"
        ]
        
        for question in test_questions:
            print(f"\n" + "="*60)
            print(f"❓ Question: {question}")
            print("-" * 60)
            
            result = agent.ask_question(question)
            
            print(f"🤖 Response: {result['response'][:100]}...")
            print(f"📊 Type: {result['question_type']}")
            print(f"📚 From dataset: {result['is_dataset_response']}")
            print(f"📊 Stage: {result['processing_stage']}")
            if result['similarity_scores']:
                print(f"🎯 Confidence: {result['similarity_scores'][0]:.3f}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 