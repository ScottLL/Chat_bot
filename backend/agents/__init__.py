"""
LangGraph agents for DeFi Q&A processing.

This package contains the agent implementations for processing DeFi-related questions
using LangGraph workflows.
"""

def __getattr__(name):
    """Lazy imports to avoid circular import issues."""
    if name == "DeFiQAAgent":
        from .defi_qa_agent import DeFiQAAgent
        return DeFiQAAgent
    elif name == "AsyncDeFiQAAgent":
        from .async_defi_qa_agent import AsyncDeFiQAAgent
        return AsyncDeFiQAAgent
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ["DeFiQAAgent", "AsyncDeFiQAAgent"] 