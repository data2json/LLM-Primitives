#!/usr/bin/env python
"""
Context Management Module

This module demonstrates context management primitives for LLMs:
- Context window management and sliding window approaches
- Relevance scoring algorithms (TF-IDF-like, semantic similarity)
- Context prioritization and compression
- Multimodal context support (text, images, audio metadata)

Educational focus: Understanding how context affects LLM behavior and responses.
"""

import json
import math
import requests
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from enum import Enum
import re


# =============================================================================
# Core Configuration
# =============================================================================

API_URL = "http://0.0.0.0:8000/v1/chat/completions"
DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


# =============================================================================
# Context Item Types
# =============================================================================

class ContextType(Enum):
    """Types of context items that can be managed."""
    TEXT = "text"
    CONVERSATION = "conversation"
    DOCUMENT = "document"
    IMAGE_DESCRIPTION = "image_description"
    AUDIO_TRANSCRIPT = "audio_transcript"
    CODE = "code"
    STRUCTURED_DATA = "structured_data"


@dataclass
class ContextItem:
    """
    A single item of context with metadata.

    Attributes:
        content: The actual content (text, description, etc.)
        context_type: The type of context
        timestamp: When this context was added
        source: Where this context came from
        relevance_score: Computed relevance (0-1)
        token_estimate: Estimated token count
        metadata: Additional contextual information
    """
    content: str
    context_type: ContextType = ContextType.TEXT
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "user"
    relevance_score: float = 1.0
    token_estimate: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.token_estimate == 0:
            # Rough estimation: ~4 chars per token for English
            self.token_estimate = len(self.content) // 4


@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)


# =============================================================================
# Relevance Scoring Algorithms
# =============================================================================

class RelevanceScorer(ABC):
    """Abstract base class for relevance scoring algorithms."""

    @abstractmethod
    def score(self, query: str, context_item: ContextItem) -> float:
        """Score the relevance of a context item to a query (0-1)."""
        pass


class KeywordRelevanceScorer(RelevanceScorer):
    """
    Simple keyword-based relevance scoring.

    Uses term frequency to measure relevance.
    """

    def _tokenize(self, text: str) -> list:
        """Simple tokenization: lowercase and split on non-alphanumeric."""
        return re.findall(r'\b\w+\b', text.lower())

    def score(self, query: str, context_item: ContextItem) -> float:
        query_tokens = set(self._tokenize(query))
        content_tokens = self._tokenize(context_item.content)

        if not query_tokens or not content_tokens:
            return 0.0

        # Count matching tokens
        content_counter = Counter(content_tokens)
        matches = sum(content_counter[token] for token in query_tokens)

        # Normalize by content length
        return min(1.0, matches / (len(content_tokens) * 0.5))


class TFIDFRelevanceScorer(RelevanceScorer):
    """
    TF-IDF inspired relevance scoring.

    Weights terms by their frequency in the item vs. corpus.
    """

    def __init__(self):
        self.document_frequencies = Counter()
        self.total_documents = 0

    def _tokenize(self, text: str) -> list:
        return re.findall(r'\b\w+\b', text.lower())

    def add_document(self, text: str):
        """Add a document to the corpus for IDF calculation."""
        tokens = set(self._tokenize(text))
        for token in tokens:
            self.document_frequencies[token] += 1
        self.total_documents += 1

    def score(self, query: str, context_item: ContextItem) -> float:
        query_tokens = self._tokenize(query)
        content_tokens = self._tokenize(context_item.content)

        if not query_tokens or not content_tokens:
            return 0.0

        content_counter = Counter(content_tokens)
        content_length = len(content_tokens)

        total_score = 0.0
        for token in query_tokens:
            # Term frequency in document
            tf = content_counter[token] / content_length if content_length > 0 else 0

            # Inverse document frequency
            df = self.document_frequencies.get(token, 0)
            idf = math.log((self.total_documents + 1) / (df + 1)) + 1 if self.total_documents > 0 else 1

            total_score += tf * idf

        # Normalize
        return min(1.0, total_score / len(query_tokens)) if query_tokens else 0.0


class RecencyRelevanceScorer(RelevanceScorer):
    """
    Scores relevance based on recency.

    More recent items get higher scores.
    """

    def __init__(self, decay_hours: float = 24.0):
        self.decay_hours = decay_hours

    def score(self, query: str, context_item: ContextItem) -> float:
        age = datetime.now() - context_item.timestamp
        age_hours = age.total_seconds() / 3600

        # Exponential decay
        return math.exp(-age_hours / self.decay_hours)


class CompositeRelevanceScorer(RelevanceScorer):
    """
    Combines multiple relevance scorers with weights.
    """

    def __init__(self, scorers: list = None):
        """
        Args:
            scorers: List of (scorer, weight) tuples
        """
        self.scorers = scorers or []

    def add_scorer(self, scorer: RelevanceScorer, weight: float = 1.0):
        self.scorers.append((scorer, weight))

    def score(self, query: str, context_item: ContextItem) -> float:
        if not self.scorers:
            return 0.0

        total_weight = sum(w for _, w in self.scorers)
        weighted_sum = sum(
            scorer.score(query, context_item) * weight
            for scorer, weight in self.scorers
        )

        return weighted_sum / total_weight if total_weight > 0 else 0.0


# =============================================================================
# Context Window Manager
# =============================================================================

class ContextWindow:
    """
    Manages a fixed-size context window for LLM interactions.

    Features:
    - Fixed token budget management
    - Relevance-based prioritization
    - Sliding window for conversations
    - Context compression strategies
    """

    def __init__(self, max_tokens: int = 4096,
                 scorer: RelevanceScorer = None):
        self.max_tokens = max_tokens
        self.scorer = scorer or KeywordRelevanceScorer()
        self.items: list = []
        self.conversation_history: list = []
        self.system_prompt: Optional[str] = None

    def set_system_prompt(self, prompt: str):
        """Set the system prompt (always included in context)."""
        self.system_prompt = prompt

    def add_item(self, item: ContextItem):
        """Add a context item to the window."""
        self.items.append(item)

    def add_conversation_turn(self, role: str, content: str, **metadata):
        """Add a conversation turn."""
        turn = ConversationTurn(role=role, content=content, metadata=metadata)
        self.conversation_history.append(turn)

    def add_text(self, text: str, context_type: ContextType = ContextType.TEXT,
                 source: str = "user", **metadata):
        """Convenience method to add text context."""
        item = ContextItem(
            content=text,
            context_type=context_type,
            source=source,
            metadata=metadata
        )
        self.add_item(item)

    def get_current_token_usage(self) -> int:
        """Calculate current token usage."""
        total = 0
        if self.system_prompt:
            total += len(self.system_prompt) // 4
        total += sum(item.token_estimate for item in self.items)
        total += sum(len(turn.content) // 4 for turn in self.conversation_history)
        return total

    def score_items(self, query: str):
        """Score all items against a query."""
        for item in self.items:
            item.relevance_score = self.scorer.score(query, item)

    def get_relevant_context(self, query: str, max_items: int = None) -> list:
        """Get items sorted by relevance to query."""
        self.score_items(query)
        sorted_items = sorted(self.items, key=lambda x: x.relevance_score, reverse=True)
        if max_items:
            return sorted_items[:max_items]
        return sorted_items

    def build_context_string(self, query: str,
                             include_conversation: bool = True,
                             max_context_tokens: int = None) -> str:
        """
        Build a context string optimized for the query.

        Args:
            query: The current query/instruction
            include_conversation: Whether to include conversation history
            max_context_tokens: Override for max tokens

        Returns:
            Formatted context string
        """
        max_tokens = max_context_tokens or self.max_tokens
        parts = []
        current_tokens = 0

        # System prompt always first
        if self.system_prompt:
            parts.append(f"[System Context]\n{self.system_prompt}")
            current_tokens += len(self.system_prompt) // 4

        # Add relevant context items
        relevant_items = self.get_relevant_context(query)
        context_section = []

        for item in relevant_items:
            if current_tokens + item.token_estimate > max_tokens * 0.7:
                break
            context_section.append(
                f"[{item.context_type.value.upper()} | "
                f"relevance: {item.relevance_score:.2f}]\n{item.content}"
            )
            current_tokens += item.token_estimate

        if context_section:
            parts.append("[Retrieved Context]\n" + "\n\n".join(context_section))

        # Add conversation history (most recent first, within budget)
        if include_conversation and self.conversation_history:
            conv_section = []
            for turn in reversed(self.conversation_history[-10:]):  # Last 10 turns
                turn_tokens = len(turn.content) // 4
                if current_tokens + turn_tokens > max_tokens * 0.9:
                    break
                conv_section.insert(0, f"{turn.role.upper()}: {turn.content}")
                current_tokens += turn_tokens

            if conv_section:
                parts.append("[Conversation History]\n" + "\n".join(conv_section))

        return "\n\n".join(parts)

    def compress_context(self, strategy: str = "truncate") -> int:
        """
        Compress context to fit within token budget.

        Strategies:
        - truncate: Remove oldest/least relevant items
        - summarize: (Would require LLM call) Replace with summaries

        Returns:
            Number of items removed
        """
        removed = 0

        if strategy == "truncate":
            # Remove items with lowest relevance
            while self.get_current_token_usage() > self.max_tokens and self.items:
                # Sort by relevance and remove lowest
                self.items.sort(key=lambda x: x.relevance_score)
                if self.items:
                    self.items.pop(0)
                    removed += 1

        return removed

    def clear(self):
        """Clear all context."""
        self.items.clear()
        self.conversation_history.clear()

    def get_stats(self) -> dict:
        """Get statistics about the current context window."""
        return {
            "total_items": len(self.items),
            "conversation_turns": len(self.conversation_history),
            "estimated_tokens": self.get_current_token_usage(),
            "max_tokens": self.max_tokens,
            "utilization": self.get_current_token_usage() / self.max_tokens,
            "context_types": Counter(item.context_type.value for item in self.items)
        }


# =============================================================================
# Multimodal Context Support
# =============================================================================

class MultimodalContextManager:
    """
    Manages context from multiple modalities.

    While LLMs primarily process text, this class handles:
    - Image descriptions/captions
    - Audio transcripts
    - Structured data representations
    """

    def __init__(self, context_window: ContextWindow = None):
        self.context_window = context_window or ContextWindow()

    def add_image_context(self, description: str, source: str = "image",
                          image_path: str = None, **metadata):
        """Add context from an image (as description)."""
        meta = {"image_path": image_path, **metadata}
        self.context_window.add_text(
            f"[Image Description]: {description}",
            context_type=ContextType.IMAGE_DESCRIPTION,
            source=source,
            **meta
        )

    def add_audio_context(self, transcript: str, source: str = "audio",
                          duration_seconds: float = None, **metadata):
        """Add context from audio (as transcript)."""
        meta = {"duration_seconds": duration_seconds, **metadata}
        self.context_window.add_text(
            f"[Audio Transcript]: {transcript}",
            context_type=ContextType.AUDIO_TRANSCRIPT,
            source=source,
            **meta
        )

    def add_code_context(self, code: str, language: str = "python",
                         file_path: str = None, **metadata):
        """Add code context."""
        meta = {"language": language, "file_path": file_path, **metadata}
        formatted = f"[Code ({language})]:\n```{language}\n{code}\n```"
        self.context_window.add_text(
            formatted,
            context_type=ContextType.CODE,
            source=file_path or "code",
            **meta
        )

    def add_structured_data(self, data: dict, description: str = "",
                            source: str = "data"):
        """Add structured data as JSON context."""
        formatted = f"[Structured Data]: {description}\n{json.dumps(data, indent=2)}"
        self.context_window.add_text(
            formatted,
            context_type=ContextType.STRUCTURED_DATA,
            source=source
        )


# =============================================================================
# Context-Aware LLM Interface
# =============================================================================

class ContextAwareLLM:
    """
    LLM interface with integrated context management.
    """

    def __init__(self, api_url: str = API_URL, model: str = DEFAULT_MODEL,
                 max_context_tokens: int = 4096):
        self.api_url = api_url
        self.model = model
        self.context_manager = MultimodalContextManager(
            ContextWindow(max_tokens=max_context_tokens)
        )

    def add_context(self, content: str, context_type: ContextType = ContextType.TEXT):
        """Add context for future queries."""
        self.context_manager.context_window.add_text(content, context_type)

    def query(self, prompt: str, temperature: float = 0.7,
              max_tokens: int = 1024, use_context: bool = True) -> dict:
        """
        Query the LLM with context-aware prompting.

        Returns:
            Dict with response and context metadata
        """
        context_window = self.context_manager.context_window

        # Build context-enhanced prompt
        if use_context:
            context_string = context_window.build_context_string(prompt)
            full_prompt = f"{context_string}\n\n[Current Query]\n{prompt}"
        else:
            full_prompt = prompt

        # Make API call
        messages = [{"role": "user", "content": full_prompt}]

        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                json=data
            )
            response.raise_for_status()
            llm_response = response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            llm_response = f"[API Error: {e}]"

        # Add to conversation history
        context_window.add_conversation_turn("user", prompt)
        context_window.add_conversation_turn("assistant", llm_response)

        return {
            "response": llm_response,
            "context_stats": context_window.get_stats(),
            "prompt_length": len(full_prompt)
        }


# =============================================================================
# Demonstration Functions
# =============================================================================

def demo_relevance_scoring():
    """Demonstrate relevance scoring algorithms."""
    print("=" * 70)
    print("DEMO: Relevance Scoring Algorithms")
    print("=" * 70)

    # Create test context items
    items = [
        ContextItem("Python is a programming language used for web development and data science"),
        ContextItem("JavaScript runs in web browsers and enables interactive websites"),
        ContextItem("Machine learning models can classify images and generate text"),
        ContextItem("Docker containers package applications with their dependencies"),
    ]

    query = "How do I use Python for machine learning?"

    # Test different scorers
    scorers = [
        ("Keyword", KeywordRelevanceScorer()),
        ("TF-IDF", TFIDFRelevanceScorer()),
    ]

    for name, scorer in scorers:
        print(f"\n--- {name} Scorer ---")
        if hasattr(scorer, 'add_document'):
            for item in items:
                scorer.add_document(item.content)

        for item in items:
            score = scorer.score(query, item)
            print(f"Score: {score:.3f} | {item.content[:50]}...")


def demo_context_window():
    """Demonstrate context window management."""
    print("\n" + "=" * 70)
    print("DEMO: Context Window Management")
    print("=" * 70)

    window = ContextWindow(max_tokens=500)

    # Add various context
    window.set_system_prompt("You are a helpful coding assistant.")

    window.add_text(
        "The user is working on a Flask web application.",
        context_type=ContextType.TEXT
    )
    window.add_text(
        "Previous error: ImportError - flask module not found",
        context_type=ContextType.TEXT
    )
    window.add_text(
        "def hello():\n    return 'Hello World'",
        context_type=ContextType.CODE
    )

    # Add conversation
    window.add_conversation_turn("user", "How do I install Flask?")
    window.add_conversation_turn("assistant", "You can install Flask using pip: pip install flask")
    window.add_conversation_turn("user", "I'm getting import errors")

    # Build context for new query
    query = "Why can't Python find the flask module?"
    context = window.build_context_string(query)

    print(f"\nBuilt Context for query: '{query}'")
    print("-" * 50)
    print(context[:1000])

    print("\n--- Stats ---")
    stats = window.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")


def demo_multimodal():
    """Demonstrate multimodal context management."""
    print("\n" + "=" * 70)
    print("DEMO: Multimodal Context Management")
    print("=" * 70)

    manager = MultimodalContextManager()

    # Add different types of context
    manager.add_image_context(
        "A flowchart showing user authentication process with login, validation, and session creation steps",
        image_path="/docs/auth_flow.png"
    )

    manager.add_audio_context(
        "The user mentioned they want to add two-factor authentication to the login process",
        duration_seconds=45.0
    )

    manager.add_code_context(
        """def login(username, password):
    user = User.query.filter_by(username=username).first()
    if user and user.check_password(password):
        return create_session(user)
    return None""",
        language="python",
        file_path="auth.py"
    )

    manager.add_structured_data(
        {"users": 1500, "active_sessions": 234, "failed_logins_24h": 12},
        description="Current authentication metrics"
    )

    # Show stats
    print("\n--- Multimodal Context Stats ---")
    stats = manager.context_window.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

    # Build context
    query = "How should I implement 2FA?"
    context = manager.context_window.build_context_string(query)
    print(f"\n--- Context for '{query}' ---")
    print(context)


def demo_with_api():
    """
    Demonstrate context-aware LLM queries.

    Requires a running vLLM server.
    """
    print("\n" + "=" * 70)
    print("DEMO: Context-Aware LLM Queries")
    print("=" * 70)

    llm = ContextAwareLLM(max_context_tokens=2048)

    # Add some context
    llm.add_context(
        "The project uses Python 3.10 with FastAPI for the backend.",
        ContextType.TEXT
    )
    llm.add_context(
        "Database is PostgreSQL with SQLAlchemy ORM.",
        ContextType.TEXT
    )

    # Query with context
    result = llm.query("What's the best way to handle database migrations?")

    print(f"\nResponse:\n{result['response'][:500]}...")
    print(f"\n--- Context Stats ---")
    for key, value in result['context_stats'].items():
        print(f"{key}: {value}")


def main():
    """Main entry point demonstrating the Context Management Module."""
    print("\n" + "=" * 70)
    print("CONTEXT MANAGEMENT MODULE - LLM Primitives")
    print("=" * 70)

    # Demos that work without API
    demo_relevance_scoring()
    demo_context_window()
    demo_multimodal()

    # Demo with API
    print("\n" + "=" * 70)
    print("API DEMO (requires running vLLM server)")
    print("=" * 70)

    try:
        demo_with_api()
    except Exception as e:
        print(f"\nNote: API demo skipped - {e}")
        print("Start a vLLM server to run API-dependent demos.")


if __name__ == "__main__":
    main()
