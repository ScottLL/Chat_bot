"""
Enhanced error handling system for the DeFi Q&A API.

This module provides structured error responses with categorization,
helpful guidance, and consistent error formatting across the application.
"""

from enum import Enum
from typing import Dict, Any, Optional, List
from pydantic import BaseModel


class ErrorCategory(str, Enum):
    """Error categories for better error handling and user guidance."""
    
    VALIDATION = "validation"
    NETWORK = "network"
    API_ERROR = "api_error"
    NO_RESULTS = "no_results"
    PROCESSING = "processing"
    SYSTEM = "system"
    RATE_LIMIT = "rate_limit"


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class UserGuidance(BaseModel):
    """User guidance for error recovery."""
    
    title: str
    suggestions: List[str]
    can_retry: bool = True
    retry_delay: Optional[int] = None  # seconds


class StructuredError(BaseModel):
    """Structured error response with enhanced information."""
    
    error: str
    category: ErrorCategory
    severity: ErrorSeverity
    user_guidance: UserGuidance
    technical_details: Optional[str] = None
    processing_stage: Optional[str] = None


class ErrorHandler:
    """Enhanced error handler with categorization and user guidance."""
    
    @staticmethod
    def validation_error(message: str, field: Optional[str] = None) -> StructuredError:
        """Handle validation errors."""
        guidance_suggestions = [
            "Check that your question is between 3-500 characters",
            "Make sure your question contains only valid text",
            "Try rephrasing your question"
        ]
        
        if field:
            guidance_suggestions.insert(0, f"Check the '{field}' field in your request")
            
        return StructuredError(
            error=message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            user_guidance=UserGuidance(
                title="Input Validation Issue",
                suggestions=guidance_suggestions,
                can_retry=True
            ),
            processing_stage="validation"
        )
    
    @staticmethod
    def no_results_error(query: str) -> StructuredError:
        """Handle no results found errors."""
        return StructuredError(
            error="No relevant answers found for your question",
            category=ErrorCategory.NO_RESULTS,
            severity=ErrorSeverity.MEDIUM,
            user_guidance=UserGuidance(
                title="No Match Found",
                suggestions=[
                    "Try rephrasing your question",
                    "Ask about DeFi topics like lending, staking, or yield farming",
                    "Use more specific terms related to DeFi protocols",
                    "Include protocol names like Aave, Uniswap, or Compound",
                    "Ask about concepts like APY, liquidity pools, or smart contracts"
                ],
                can_retry=True
            ),
            processing_stage="semantic_search"
        )
    
    @staticmethod
    def api_error(message: str, provider: str = "OpenAI") -> StructuredError:
        """Handle external API errors."""
        return StructuredError(
            error=f"{provider} API error: {message}",
            category=ErrorCategory.API_ERROR,
            severity=ErrorSeverity.HIGH,
            user_guidance=UserGuidance(
                title="External Service Issue",
                suggestions=[
                    "This is a temporary service issue",
                    "Please try again in a few moments",
                    "If the issue persists, contact support"
                ],
                can_retry=True,
                retry_delay=5
            ),
            processing_stage="external_api",
            technical_details=message
        )
    
    @staticmethod
    def rate_limit_error(retry_after: Optional[int] = None) -> StructuredError:
        """Handle rate limiting errors."""
        suggestions = [
            "You've sent too many requests recently",
            "Please wait before sending another question"
        ]
        
        if retry_after:
            suggestions.append(f"Try again in {retry_after} seconds")
        else:
            suggestions.append("Try again in a few moments")
            
        return StructuredError(
            error="Too many requests - please slow down",
            category=ErrorCategory.RATE_LIMIT,
            severity=ErrorSeverity.MEDIUM,
            user_guidance=UserGuidance(
                title="Rate Limit Exceeded",
                suggestions=suggestions,
                can_retry=True,
                retry_delay=retry_after or 10
            ),
            processing_stage="rate_limiting"
        )
    
    @staticmethod
    def network_error(message: str) -> StructuredError:
        """Handle network connectivity errors."""
        return StructuredError(
            error=f"Network error: {message}",
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.HIGH,
            user_guidance=UserGuidance(
                title="Connection Issue",
                suggestions=[
                    "Check your internet connection",
                    "Verify the API server is running",
                    "Try refreshing the page",
                    "Contact support if the issue continues"
                ],
                can_retry=True,
                retry_delay=3
            ),
            processing_stage="network",
            technical_details=message
        )
    
    @staticmethod
    def processing_error(message: str, stage: str = "unknown") -> StructuredError:
        """Handle general processing errors."""
        return StructuredError(
            error=f"Processing error: {message}",
            category=ErrorCategory.PROCESSING,
            severity=ErrorSeverity.MEDIUM,
            user_guidance=UserGuidance(
                title="Processing Issue",
                suggestions=[
                    "There was an issue processing your request",
                    "Try rephrasing your question",
                    "If the error persists, contact support"
                ],
                can_retry=True
            ),
            processing_stage=stage,
            technical_details=message
        )
    
    @staticmethod
    def system_error(message: str) -> StructuredError:
        """Handle critical system errors."""
        return StructuredError(
            error="System temporarily unavailable",
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            user_guidance=UserGuidance(
                title="System Issue",
                suggestions=[
                    "The service is temporarily unavailable",
                    "Please try again later",
                    "Contact support if the issue persists"
                ],
                can_retry=True,
                retry_delay=30
            ),
            processing_stage="system",
            technical_details=message
        )
    
    @staticmethod
    def agent_unavailable_error() -> StructuredError:
        """Handle agent not loaded errors."""
        return StructuredError(
            error="DeFi Q&A service is currently unavailable",
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            user_guidance=UserGuidance(
                title="Service Unavailable",
                suggestions=[
                    "The Q&A service is starting up",
                    "Please wait a moment and try again",
                    "Refresh the page if the issue persists",
                    "Contact support if the service remains unavailable"
                ],
                can_retry=True,
                retry_delay=10
            ),
            processing_stage="agent_initialization"
        )
    
    @staticmethod
    def configuration_error(message: str) -> StructuredError:
        """Handle configuration errors (e.g., missing API keys)."""
        return StructuredError(
            error=message,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            user_guidance=UserGuidance(
                title="Configuration Issue",
                suggestions=[
                    "The service is not properly configured",
                    "Contact the administrator to set up the required API keys",
                    "Check that all environment variables are properly set"
                ],
                can_retry=False
            ),
            processing_stage="configuration"
        )


def format_error_for_response(error: StructuredError) -> Dict[str, Any]:
    """Format structured error for API response."""
    return {
        "error": error.error,
        "category": error.category.value,
        "severity": error.severity.value,
        "user_guidance": {
            "title": error.user_guidance.title,
            "suggestions": error.user_guidance.suggestions,
            "can_retry": error.user_guidance.can_retry,
            "retry_delay": error.user_guidance.retry_delay
        },
        "processing_stage": error.processing_stage,
        "technical_details": error.technical_details
    }


def format_error_for_stream(error: StructuredError) -> Dict[str, Any]:
    """Format structured error for streaming response."""
    return {
        "type": "error",
        "error": error.error,
        "category": error.category.value,
        "user_guidance": error.user_guidance.dict(),
        "processing_stage": error.processing_stage
    } 