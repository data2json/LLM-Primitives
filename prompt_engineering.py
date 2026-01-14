#!/usr/bin/env python
"""
Prompt Engineering Module

This module demonstrates fundamental prompt engineering primitives for LLMs:
- Dynamic prompt templating based on conversation flow
- Multiple prompt strategies (multi-perspective, chain-of-thought, iterative refinement)
- Template composition and variable substitution

Educational focus: Understanding how prompt structure affects LLM outputs.
"""

import json
import requests
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


# =============================================================================
# Core Configuration
# =============================================================================

API_URL = "http://0.0.0.0:8000/v1/chat/completions"
DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


# =============================================================================
# Prompt Templates
# =============================================================================

@dataclass
class PromptTemplate:
    """
    A dynamic prompt template with variable substitution.

    Templates use {variable_name} syntax for substitution.
    Example: "Analyze {topic} from the perspective of {role}"
    """
    template: str
    name: str = "unnamed"
    description: str = ""
    required_vars: list = field(default_factory=list)

    def render(self, **kwargs) -> str:
        """Render the template with provided variables."""
        missing = [v for v in self.required_vars if v not in kwargs]
        if missing:
            raise ValueError(f"Missing required variables: {missing}")
        return self.template.format(**kwargs)

    def __add__(self, other: "PromptTemplate") -> "PromptTemplate":
        """Compose two templates together."""
        return PromptTemplate(
            template=f"{self.template}\n\n{other.template}",
            name=f"{self.name}+{other.name}",
            required_vars=list(set(self.required_vars + other.required_vars))
        )


# =============================================================================
# Built-in Template Library
# =============================================================================

TEMPLATES = {
    "basic": PromptTemplate(
        template="{instruction}",
        name="basic",
        description="Simple direct instruction",
        required_vars=["instruction"]
    ),

    "role_based": PromptTemplate(
        template="You are {role}. {instruction}",
        name="role_based",
        description="Assigns a role/persona before the instruction",
        required_vars=["role", "instruction"]
    ),

    "chain_of_thought": PromptTemplate(
        template="""{instruction}

Let's think through this step by step:
1. First, identify the key elements of the problem
2. Consider relevant factors and constraints
3. Analyze possible approaches
4. Arrive at a well-reasoned conclusion""",
        name="chain_of_thought",
        description="Encourages step-by-step reasoning",
        required_vars=["instruction"]
    ),

    "multi_perspective": PromptTemplate(
        template="""{instruction}

Analyze this from multiple perspectives:
- Technical perspective: What are the practical implications?
- Ethical perspective: What are the moral considerations?
- User perspective: How does this affect end users?
- Long-term perspective: What are the future implications?

Synthesize these viewpoints into a balanced response.""",
        name="multi_perspective",
        description="Forces consideration of multiple viewpoints",
        required_vars=["instruction"]
    ),

    "socratic": PromptTemplate(
        template="""{instruction}

Before answering, consider these guiding questions:
- What assumptions am I making?
- What evidence supports different conclusions?
- What are the potential counterarguments?
- How confident am I in this analysis?

Now provide a thoughtful response.""",
        name="socratic",
        description="Uses Socratic questioning to deepen analysis",
        required_vars=["instruction"]
    ),

    "structured_output": PromptTemplate(
        template="""{instruction}

Provide your response in the following format:
## Summary
[Brief overview]

## Analysis
[Detailed analysis]

## Recommendations
[Actionable items]

## Confidence Level
[Low/Medium/High with justification]""",
        name="structured_output",
        description="Requests structured formatted output",
        required_vars=["instruction"]
    ),

    "few_shot": PromptTemplate(
        template="""Here are some examples:

{examples}

Now, following the same pattern:
{instruction}""",
        name="few_shot",
        description="Provides examples before the task",
        required_vars=["examples", "instruction"]
    ),

    "constraint_based": PromptTemplate(
        template="""{instruction}

Constraints:
{constraints}

Ensure your response adheres to all specified constraints.""",
        name="constraint_based",
        description="Adds explicit constraints to the task",
        required_vars=["instruction", "constraints"]
    ),
}


# =============================================================================
# Prompt Strategies (Abstract Pattern)
# =============================================================================

class PromptStrategy(ABC):
    """
    Abstract base class for prompt strategies.

    A strategy defines how prompts are constructed and potentially
    how multiple LLM calls are orchestrated.
    """

    @abstractmethod
    def build_prompt(self, context: dict) -> str:
        """Build the prompt from context."""
        pass

    @abstractmethod
    def process_response(self, response: str, context: dict) -> dict:
        """Process the LLM response and return structured output."""
        pass


class DirectStrategy(PromptStrategy):
    """Simple direct prompting without any modifications."""

    def __init__(self, template: PromptTemplate = None):
        self.template = template or TEMPLATES["basic"]

    def build_prompt(self, context: dict) -> str:
        return self.template.render(**context)

    def process_response(self, response: str, context: dict) -> dict:
        return {"response": response, "strategy": "direct"}


class ChainOfThoughtStrategy(PromptStrategy):
    """Encourages step-by-step reasoning in responses."""

    def __init__(self):
        self.template = TEMPLATES["chain_of_thought"]

    def build_prompt(self, context: dict) -> str:
        return self.template.render(**context)

    def process_response(self, response: str, context: dict) -> dict:
        # Attempt to extract steps from the response
        lines = response.strip().split("\n")
        steps = [l.strip() for l in lines if l.strip().startswith(("1.", "2.", "3.", "4.", "5.", "-", "*"))]
        return {
            "response": response,
            "strategy": "chain_of_thought",
            "extracted_steps": steps
        }


class MultiPerspectiveStrategy(PromptStrategy):
    """Generates responses from multiple viewpoints."""

    def __init__(self, perspectives: list = None):
        self.perspectives = perspectives or [
            "technical", "ethical", "user-focused", "long-term"
        ]
        self.template = TEMPLATES["multi_perspective"]

    def build_prompt(self, context: dict) -> str:
        return self.template.render(**context)

    def process_response(self, response: str, context: dict) -> dict:
        return {
            "response": response,
            "strategy": "multi_perspective",
            "perspectives_requested": self.perspectives
        }


class IterativeRefinementStrategy(PromptStrategy):
    """
    Implements iterative refinement through multiple LLM calls.

    This strategy:
    1. Gets an initial response
    2. Critiques the response
    3. Refines based on critique
    """

    def __init__(self, max_iterations: int = 2):
        self.max_iterations = max_iterations
        self.initial_template = TEMPLATES["basic"]
        self.critique_template = PromptTemplate(
            template="""Review this response and identify areas for improvement:

Response to review:
{response}

Original question: {instruction}

Provide specific, constructive feedback for improvement.""",
            name="critique",
            required_vars=["response", "instruction"]
        )
        self.refine_template = PromptTemplate(
            template="""Improve this response based on the feedback:

Original response:
{response}

Feedback:
{critique}

Original question: {instruction}

Provide an improved response addressing the feedback.""",
            name="refine",
            required_vars=["response", "critique", "instruction"]
        )

    def build_prompt(self, context: dict) -> str:
        return self.initial_template.render(**context)

    def process_response(self, response: str, context: dict) -> dict:
        return {
            "response": response,
            "strategy": "iterative_refinement",
            "iteration": context.get("iteration", 0)
        }

    def get_critique_prompt(self, response: str, instruction: str) -> str:
        return self.critique_template.render(response=response, instruction=instruction)

    def get_refine_prompt(self, response: str, critique: str, instruction: str) -> str:
        return self.refine_template.render(response=response, critique=critique, instruction=instruction)


# =============================================================================
# Prompt Manager (Orchestrator)
# =============================================================================

class PromptManager:
    """
    Central manager for prompt engineering operations.

    Handles template selection, strategy application, and LLM communication.
    """

    def __init__(self, api_url: str = API_URL, model: str = DEFAULT_MODEL):
        self.api_url = api_url
        self.model = model
        self.templates = TEMPLATES.copy()
        self.conversation_history = []

    def register_template(self, name: str, template: PromptTemplate):
        """Register a custom template."""
        self.templates[name] = template

    def make_api_call(self, prompt: str, system_prompt: str = None,
                      temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """Make an API call to the LLM."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

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
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            return f"[API Error: {e}]"

    def execute_strategy(self, strategy: PromptStrategy, context: dict,
                        system_prompt: str = None) -> dict:
        """Execute a prompt strategy and return results."""
        prompt = strategy.build_prompt(context)
        response = self.make_api_call(prompt, system_prompt)
        result = strategy.process_response(response, context)

        # Track in conversation history
        self.conversation_history.append({
            "prompt": prompt,
            "response": response,
            "strategy": type(strategy).__name__
        })

        return result

    def execute_iterative_refinement(self, instruction: str,
                                     max_iterations: int = 2) -> dict:
        """
        Execute the iterative refinement strategy with multiple LLM calls.

        Returns the evolution of responses through refinement.
        """
        strategy = IterativeRefinementStrategy(max_iterations)

        # Initial response
        initial_prompt = strategy.build_prompt({"instruction": instruction})
        current_response = self.make_api_call(initial_prompt)

        iterations = [{
            "iteration": 0,
            "type": "initial",
            "response": current_response
        }]

        # Refinement iterations
        for i in range(max_iterations):
            # Get critique
            critique_prompt = strategy.get_critique_prompt(current_response, instruction)
            critique = self.make_api_call(critique_prompt)

            iterations.append({
                "iteration": i + 1,
                "type": "critique",
                "response": critique
            })

            # Get refined response
            refine_prompt = strategy.get_refine_prompt(current_response, critique, instruction)
            current_response = self.make_api_call(refine_prompt)

            iterations.append({
                "iteration": i + 1,
                "type": "refinement",
                "response": current_response
            })

        return {
            "final_response": current_response,
            "iterations": iterations,
            "total_llm_calls": 1 + (max_iterations * 2)
        }


# =============================================================================
# Demonstration Functions
# =============================================================================

def demo_template_composition():
    """Demonstrate template composition and rendering."""
    print("=" * 70)
    print("DEMO: Template Composition")
    print("=" * 70)

    # Create custom templates
    context_template = PromptTemplate(
        template="Context: {context}",
        name="context",
        required_vars=["context"]
    )

    task_template = PromptTemplate(
        template="Task: {task}",
        name="task",
        required_vars=["task"]
    )

    # Compose templates
    combined = context_template + task_template

    # Render
    result = combined.render(
        context="You are helping debug a Python application",
        task="Explain why this function might cause a memory leak"
    )

    print(f"Combined template:\n{result}\n")


def demo_strategies():
    """Demonstrate different prompt strategies (without API calls)."""
    print("=" * 70)
    print("DEMO: Prompt Strategies")
    print("=" * 70)

    instruction = "Explain the trade-offs of using microservices vs monolithic architecture"

    strategies = [
        ("Direct", DirectStrategy()),
        ("Chain of Thought", ChainOfThoughtStrategy()),
        ("Multi-Perspective", MultiPerspectiveStrategy()),
    ]

    for name, strategy in strategies:
        prompt = strategy.build_prompt({"instruction": instruction})
        print(f"\n--- {name} Strategy ---")
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        print()


def demo_with_api():
    """
    Demonstrate prompt engineering with actual API calls.

    Requires a running vLLM server.
    """
    print("=" * 70)
    print("DEMO: Prompt Engineering with API")
    print("=" * 70)

    manager = PromptManager()

    instruction = "What are the key considerations when designing a REST API?"

    # Test different strategies
    print("\n--- Direct Strategy ---")
    result = manager.execute_strategy(
        DirectStrategy(),
        {"instruction": instruction}
    )
    print(f"Response:\n{result['response'][:500]}...")

    print("\n--- Chain of Thought Strategy ---")
    result = manager.execute_strategy(
        ChainOfThoughtStrategy(),
        {"instruction": instruction}
    )
    print(f"Response:\n{result['response'][:500]}...")
    print(f"Extracted steps: {len(result.get('extracted_steps', []))} found")


def demo_iterative_refinement():
    """
    Demonstrate iterative refinement with multiple LLM calls.

    Requires a running vLLM server.
    """
    print("=" * 70)
    print("DEMO: Iterative Refinement")
    print("=" * 70)

    manager = PromptManager()

    instruction = "Write a brief explanation of how neural networks learn"

    result = manager.execute_iterative_refinement(instruction, max_iterations=1)

    print(f"\nTotal LLM calls made: {result['total_llm_calls']}")
    print("\n--- Refinement Evolution ---")

    for item in result["iterations"]:
        print(f"\n[Iteration {item['iteration']} - {item['type'].upper()}]")
        print(item["response"][:300] + "..." if len(item["response"]) > 300 else item["response"])

    print("\n--- Final Response ---")
    print(result["final_response"])


def main():
    """Main entry point demonstrating the Prompt Engineering Module."""
    print("\n" + "=" * 70)
    print("PROMPT ENGINEERING MODULE - LLM Primitives")
    print("=" * 70)

    # Demo without API (always works)
    demo_template_composition()
    demo_strategies()

    # Demo with API (requires vLLM server)
    print("\n" + "=" * 70)
    print("API DEMOS (requires running vLLM server)")
    print("=" * 70)

    try:
        demo_with_api()
        demo_iterative_refinement()
    except Exception as e:
        print(f"\nNote: API demos skipped - {e}")
        print("Start a vLLM server to run API-dependent demos.")


if __name__ == "__main__":
    main()
