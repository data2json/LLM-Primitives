#!/usr/bin/env python
"""
Metacognition Module

This module demonstrates metacognitive primitives for LLMs:
- Self-reflection and self-assessment
- Confidence estimation and calibration
- Adaptive complexity scaling
- Strategy optimization and learning
- Error detection and correction

Educational focus: Understanding how LLMs can reason about their own reasoning.
"""

import json
import math
import requests
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Callable
from enum import Enum


# =============================================================================
# Core Configuration
# =============================================================================

API_URL = "http://0.0.0.0:8000/v1/chat/completions"
DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


# =============================================================================
# Metacognitive Structures
# =============================================================================

class ConfidenceLevel(Enum):
    """Discrete confidence levels."""
    VERY_LOW = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9


class ComplexityLevel(Enum):
    """Task complexity levels."""
    SIMPLE = 1
    MODERATE = 2
    COMPLEX = 3
    VERY_COMPLEX = 4
    EXPERT = 5


@dataclass
class ReflectionResult:
    """Result of a self-reflection process."""
    original_response: str
    reflection: str
    confidence: float
    identified_issues: list = field(default_factory=list)
    suggested_improvements: list = field(default_factory=list)
    revised_response: Optional[str] = None


@dataclass
class ConfidenceAssessment:
    """Detailed confidence assessment."""
    overall_confidence: float
    knowledge_confidence: float  # How well does the model know this topic?
    reasoning_confidence: float  # How confident in the reasoning process?
    completeness_confidence: float  # How complete is the response?
    factors: dict = field(default_factory=dict)
    calibration_adjustment: float = 0.0


@dataclass
class StrategyRecord:
    """Record of a strategy's performance."""
    strategy_name: str
    usage_count: int = 0
    success_count: int = 0
    total_confidence: float = 0.0
    contexts: list = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.success_count / self.usage_count if self.usage_count > 0 else 0.0

    @property
    def avg_confidence(self) -> float:
        return self.total_confidence / self.usage_count if self.usage_count > 0 else 0.0


# =============================================================================
# Self-Reflection
# =============================================================================

class SelfReflector:
    """
    Enables LLMs to reflect on their own outputs.
    """

    def __init__(self, api_url: str = API_URL, model: str = DEFAULT_MODEL):
        self.api_url = api_url
        self.model = model
        self.reflection_history = []

    def _call_llm(self, prompt: str, temperature: float = 0.3) -> str:
        """Make an LLM API call."""
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 1024,
        }

        try:
            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                json=data
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[Error: {e}]"

    def reflect(self, question: str, response: str) -> ReflectionResult:
        """
        Perform self-reflection on a response.
        """
        prompt = f"""Reflect critically on this response:

QUESTION: {question}

RESPONSE: {response}

Analyze:
1. ACCURACY: Is the information accurate? Any factual errors?
2. COMPLETENESS: Does it fully address the question?
3. CLARITY: Is it clearly expressed?
4. LOGIC: Is the reasoning sound?
5. BIASES: Any potential biases or assumptions?
6. CONFIDENCE: How confident should one be in this response (0-100%)?

Provide specific issues found and suggestions for improvement:"""

        reflection = self._call_llm(prompt)

        # Extract confidence from reflection
        confidence = self._extract_confidence(reflection)

        # Identify specific issues
        issues = self._extract_issues(reflection)

        result = ReflectionResult(
            original_response=response,
            reflection=reflection,
            confidence=confidence,
            identified_issues=issues
        )

        self.reflection_history.append(result)
        return result

    def _extract_confidence(self, reflection: str) -> float:
        """Extract confidence score from reflection text."""
        import re
        # Look for percentage patterns
        patterns = [
            r'(\d+(?:\.\d+)?)\s*%',
            r'confidence[:\s]+(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*out of\s*100'
        ]

        for pattern in patterns:
            match = re.search(pattern, reflection.lower())
            if match:
                value = float(match.group(1))
                if value > 1:
                    value = value / 100
                return min(1.0, max(0.0, value))

        return 0.5  # Default moderate confidence

    def _extract_issues(self, reflection: str) -> list:
        """Extract identified issues from reflection."""
        issues = []
        # Simple extraction based on common patterns
        issue_indicators = [
            "issue:", "problem:", "error:", "incorrect",
            "missing:", "lacks:", "should:", "could improve"
        ]

        lines = reflection.lower().split('\n')
        for line in lines:
            for indicator in issue_indicators:
                if indicator in line:
                    issues.append(line.strip())
                    break

        return issues[:5]  # Limit to top 5 issues

    def reflect_and_revise(self, question: str, response: str) -> ReflectionResult:
        """
        Reflect on a response and generate an improved version.
        """
        # First, reflect
        result = self.reflect(question, response)

        # Then, revise based on reflection
        revision_prompt = f"""Based on this reflection, provide an improved response:

ORIGINAL QUESTION: {question}

ORIGINAL RESPONSE: {response}

REFLECTION/ISSUES IDENTIFIED:
{result.reflection}

Provide a revised, improved response that addresses the identified issues:"""

        result.revised_response = self._call_llm(revision_prompt)

        return result

    def meta_reflect(self, reflections: list) -> str:
        """
        Perform meta-reflection on multiple reflection results.

        This is reflection on reflections - identifying patterns in self-assessment.
        """
        reflection_summaries = "\n".join([
            f"- Confidence: {r.confidence:.0%}, Issues: {len(r.identified_issues)}"
            for r in reflections
        ])

        prompt = f"""Analyze these reflection results to identify patterns:

REFLECTION SUMMARIES:
{reflection_summaries}

Identify:
1. Recurring issues or blind spots
2. Confidence calibration patterns (over/under confident?)
3. Areas needing systematic improvement
4. Strengths to leverage

Meta-analysis:"""

        return self._call_llm(prompt)


# =============================================================================
# Confidence Estimation
# =============================================================================

class ConfidenceEstimator:
    """
    Estimates and calibrates confidence in LLM outputs.
    """

    def __init__(self, api_url: str = API_URL, model: str = DEFAULT_MODEL):
        self.api_url = api_url
        self.model = model
        self.calibration_data = []  # (estimated, actual) pairs
        self.calibration_offset = 0.0

    def _call_llm(self, prompt: str, temperature: float = 0.3) -> str:
        """Make an LLM API call."""
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 1024,
        }

        try:
            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                json=data
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[Error: {e}]"

    def estimate_confidence(self, question: str, response: str) -> ConfidenceAssessment:
        """
        Estimate confidence in a response across multiple dimensions.
        """
        prompt = f"""Assess confidence in this response on multiple dimensions:

QUESTION: {question}
RESPONSE: {response}

Rate each dimension from 0-100:
1. KNOWLEDGE_CONFIDENCE: How well-established is the underlying knowledge?
2. REASONING_CONFIDENCE: How sound is the reasoning process?
3. COMPLETENESS_CONFIDENCE: How complete is the response?
4. OVERALL_CONFIDENCE: Overall confidence in the response?

Also identify factors affecting confidence:
- Positive factors (increasing confidence)
- Negative factors (decreasing confidence)

Provide ratings and factors:"""

        analysis = self._call_llm(prompt)

        # Parse confidence values
        knowledge_conf = self._extract_dimension(analysis, "knowledge")
        reasoning_conf = self._extract_dimension(analysis, "reasoning")
        completeness_conf = self._extract_dimension(analysis, "completeness")
        overall_conf = self._extract_dimension(analysis, "overall")

        # Apply calibration adjustment
        calibrated_overall = max(0.0, min(1.0, overall_conf + self.calibration_offset))

        return ConfidenceAssessment(
            overall_confidence=calibrated_overall,
            knowledge_confidence=knowledge_conf,
            reasoning_confidence=reasoning_conf,
            completeness_confidence=completeness_conf,
            factors={"analysis": analysis},
            calibration_adjustment=self.calibration_offset
        )

    def _extract_dimension(self, text: str, dimension: str) -> float:
        """Extract a confidence dimension from text."""
        import re
        pattern = rf'{dimension}[_\s]*confidence[:\s]*(\d+)'
        match = re.search(pattern, text.lower())
        if match:
            return float(match.group(1)) / 100
        return 0.5

    def update_calibration(self, estimated: float, actual: float):
        """
        Update calibration based on feedback.

        Args:
            estimated: The estimated confidence
            actual: The actual correctness (0 or 1)
        """
        self.calibration_data.append((estimated, actual))

        # Recalculate calibration offset
        if len(self.calibration_data) >= 5:
            avg_estimated = sum(e for e, _ in self.calibration_data) / len(self.calibration_data)
            avg_actual = sum(a for _, a in self.calibration_data) / len(self.calibration_data)
            self.calibration_offset = avg_actual - avg_estimated

    def get_calibration_stats(self) -> dict:
        """Get statistics about confidence calibration."""
        if not self.calibration_data:
            return {"status": "insufficient_data"}

        estimated = [e for e, _ in self.calibration_data]
        actual = [a for _, a in self.calibration_data]

        return {
            "sample_size": len(self.calibration_data),
            "avg_estimated_confidence": sum(estimated) / len(estimated),
            "avg_actual_accuracy": sum(actual) / len(actual),
            "calibration_offset": self.calibration_offset,
            "is_overconfident": self.calibration_offset < -0.1,
            "is_underconfident": self.calibration_offset > 0.1
        }


# =============================================================================
# Confidence-Based Branching
# =============================================================================

class ConfidenceBranching:
    """
    Makes decisions based on confidence levels.
    """

    def __init__(self, estimator: ConfidenceEstimator = None):
        self.estimator = estimator or ConfidenceEstimator()
        self.threshold_low = 0.3
        self.threshold_high = 0.7

    def branch(self, question: str, response: str,
               low_action: Callable, medium_action: Callable,
               high_action: Callable) -> Any:
        """
        Branch execution based on confidence.

        Args:
            question: The question being answered
            response: The response to assess
            low_action: Action to take if confidence is low
            medium_action: Action to take if confidence is medium
            high_action: Action to take if confidence is high
        """
        assessment = self.estimator.estimate_confidence(question, response)
        confidence = assessment.overall_confidence

        if confidence < self.threshold_low:
            return low_action(question, response, assessment)
        elif confidence > self.threshold_high:
            return high_action(question, response, assessment)
        else:
            return medium_action(question, response, assessment)

    def should_escalate(self, confidence: float) -> bool:
        """Determine if a response should be escalated for review."""
        return confidence < self.threshold_low

    def should_add_caveats(self, confidence: float) -> bool:
        """Determine if caveats should be added to a response."""
        return self.threshold_low <= confidence < self.threshold_high


# =============================================================================
# Adaptive Complexity Scaling
# =============================================================================

class ComplexityScaler:
    """
    Adapts response complexity based on context and user needs.
    """

    def __init__(self, api_url: str = API_URL, model: str = DEFAULT_MODEL):
        self.api_url = api_url
        self.model = model
        self.current_level = ComplexityLevel.MODERATE
        self.user_expertise = "intermediate"

    def _call_llm(self, prompt: str, temperature: float = 0.5) -> str:
        """Make an LLM API call."""
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 1024,
        }

        try:
            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                json=data
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[Error: {e}]"

    def assess_question_complexity(self, question: str) -> ComplexityLevel:
        """Assess the complexity level of a question."""
        prompt = f"""Assess the complexity of this question:

QUESTION: {question}

Rate the complexity (1-5):
1 = SIMPLE: Basic factual question
2 = MODERATE: Requires some explanation
3 = COMPLEX: Requires detailed analysis
4 = VERY_COMPLEX: Requires expertise and nuance
5 = EXPERT: Requires deep domain knowledge

Provide your rating (just the number 1-5) and brief justification:"""

        response = self._call_llm(prompt)

        # Extract complexity level
        import re
        match = re.search(r'[1-5]', response)
        if match:
            level = int(match.group())
            return ComplexityLevel(level)

        return ComplexityLevel.MODERATE

    def scale_response(self, response: str, target_level: ComplexityLevel) -> str:
        """
        Scale a response to match target complexity level.
        """
        level_descriptions = {
            ComplexityLevel.SIMPLE: "very simple, using basic language and short sentences",
            ComplexityLevel.MODERATE: "moderately detailed with clear explanations",
            ComplexityLevel.COMPLEX: "detailed with technical depth and examples",
            ComplexityLevel.VERY_COMPLEX: "comprehensive with nuanced analysis",
            ComplexityLevel.EXPERT: "expert-level with technical precision and caveats"
        }

        prompt = f"""Rewrite this response at a {target_level.name} level:

ORIGINAL RESPONSE:
{response}

TARGET LEVEL: {level_descriptions[target_level]}

Rewritten response:"""

        return self._call_llm(prompt)

    def adapt_to_user(self, user_feedback: str) -> ComplexityLevel:
        """
        Adapt complexity based on user feedback.
        """
        prompt = f"""Based on this user feedback, determine if response complexity should change:

USER FEEDBACK: {user_feedback}

Options:
- SIMPLER: User wants simpler explanations
- SAME: Current level is appropriate
- MORE_DETAILED: User wants more detail

What adjustment is needed (SIMPLER/SAME/MORE_DETAILED)?"""

        response = self._call_llm(prompt)

        if "simpler" in response.lower():
            new_value = max(1, self.current_level.value - 1)
            self.current_level = ComplexityLevel(new_value)
        elif "more_detailed" in response.lower() or "detailed" in response.lower():
            new_value = min(5, self.current_level.value + 1)
            self.current_level = ComplexityLevel(new_value)

        return self.current_level


# =============================================================================
# Strategy Optimization
# =============================================================================

class StrategyOptimizer:
    """
    Tracks and optimizes prompting strategies based on performance.
    """

    def __init__(self, api_url: str = API_URL, model: str = DEFAULT_MODEL):
        self.api_url = api_url
        self.model = model
        self.strategies: dict = {}  # name -> StrategyRecord
        self.context_strategy_map: dict = {}  # context_type -> best_strategy

    def _call_llm(self, prompt: str, temperature: float = 0.3) -> str:
        """Make an LLM API call."""
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 1024,
        }

        try:
            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                json=data
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[Error: {e}]"

    def register_strategy(self, name: str):
        """Register a new strategy for tracking."""
        if name not in self.strategies:
            self.strategies[name] = StrategyRecord(strategy_name=name)

    def record_usage(self, strategy_name: str, context: str,
                     success: bool, confidence: float):
        """Record a strategy usage and outcome."""
        if strategy_name not in self.strategies:
            self.register_strategy(strategy_name)

        record = self.strategies[strategy_name]
        record.usage_count += 1
        record.total_confidence += confidence
        if success:
            record.success_count += 1
        record.contexts.append(context[:100])  # Store truncated context

    def recommend_strategy(self, context: str) -> str:
        """
        Recommend the best strategy for a given context.
        """
        if not self.strategies:
            return "default"

        # Find strategies with good performance
        candidates = [
            (name, record)
            for name, record in self.strategies.items()
            if record.usage_count >= 3
        ]

        if not candidates:
            # Not enough data, use LLM to suggest
            return self._llm_recommend_strategy(context)

        # Sort by success rate, then by confidence
        candidates.sort(
            key=lambda x: (x[1].success_rate, x[1].avg_confidence),
            reverse=True
        )

        return candidates[0][0]

    def _llm_recommend_strategy(self, context: str) -> str:
        """Use LLM to recommend a strategy when data is insufficient."""
        prompt = f"""Given this context, recommend the best prompting strategy:

CONTEXT: {context}

Available strategies:
1. direct: Simple, direct prompting
2. chain_of_thought: Step-by-step reasoning
3. few_shot: Provide examples first
4. role_based: Assign a role/persona
5. structured: Request structured output

Which strategy would work best? (Provide just the strategy name):"""

        response = self._call_llm(prompt)

        # Extract strategy name
        strategies = ["direct", "chain_of_thought", "few_shot", "role_based", "structured"]
        for s in strategies:
            if s in response.lower():
                return s

        return "direct"

    def get_performance_report(self) -> dict:
        """Generate a performance report for all strategies."""
        report = {}
        for name, record in self.strategies.items():
            report[name] = {
                "usage_count": record.usage_count,
                "success_rate": f"{record.success_rate:.1%}",
                "avg_confidence": f"{record.avg_confidence:.1%}",
            }
        return report

    def suggest_improvements(self) -> str:
        """
        Analyze strategy performance and suggest improvements.
        """
        if not self.strategies:
            return "Not enough data to suggest improvements."

        report = self.get_performance_report()

        prompt = f"""Analyze this strategy performance data and suggest improvements:

PERFORMANCE DATA:
{json.dumps(report, indent=2)}

Provide:
1. Which strategies are performing well and why
2. Which strategies need improvement
3. Specific suggestions for optimization
4. Contexts where different strategies might work better

Analysis and recommendations:"""

        return self._call_llm(prompt)


# =============================================================================
# Integrated Metacognition System
# =============================================================================

class MetacognitionSystem:
    """
    Integrated system combining all metacognitive capabilities.
    """

    def __init__(self, api_url: str = API_URL, model: str = DEFAULT_MODEL):
        self.api_url = api_url
        self.model = model

        self.reflector = SelfReflector(api_url, model)
        self.confidence_estimator = ConfidenceEstimator(api_url, model)
        self.complexity_scaler = ComplexityScaler(api_url, model)
        self.strategy_optimizer = StrategyOptimizer(api_url, model)
        self.branching = ConfidenceBranching(self.confidence_estimator)

        # Register default strategies
        for strategy in ["direct", "chain_of_thought", "few_shot", "role_based"]:
            self.strategy_optimizer.register_strategy(strategy)

    def process_with_metacognition(self, question: str, response: str,
                                    auto_revise: bool = True) -> dict:
        """
        Process a response with full metacognitive analysis.
        """
        # Assess complexity
        complexity = self.complexity_scaler.assess_question_complexity(question)

        # Estimate confidence
        confidence = self.confidence_estimator.estimate_confidence(question, response)

        # Reflect on response
        if auto_revise and confidence.overall_confidence < 0.7:
            reflection = self.reflector.reflect_and_revise(question, response)
        else:
            reflection = self.reflector.reflect(question, response)

        # Determine if caveats are needed
        needs_caveats = self.branching.should_add_caveats(confidence.overall_confidence)
        should_escalate = self.branching.should_escalate(confidence.overall_confidence)

        return {
            "original_response": response,
            "complexity_level": complexity.name,
            "confidence_assessment": {
                "overall": confidence.overall_confidence,
                "knowledge": confidence.knowledge_confidence,
                "reasoning": confidence.reasoning_confidence,
                "completeness": confidence.completeness_confidence
            },
            "reflection": reflection.reflection,
            "identified_issues": reflection.identified_issues,
            "revised_response": reflection.revised_response,
            "needs_caveats": needs_caveats,
            "should_escalate": should_escalate
        }

    def learn_from_feedback(self, strategy_used: str, context: str,
                           was_successful: bool, confidence: float):
        """
        Learn from user feedback to improve future performance.
        """
        # Update strategy records
        self.strategy_optimizer.record_usage(
            strategy_used, context, was_successful, confidence
        )

        # Update confidence calibration
        actual = 1.0 if was_successful else 0.0
        self.confidence_estimator.update_calibration(confidence, actual)

    def get_self_assessment(self) -> str:
        """
        Generate a self-assessment of the system's performance.
        """
        # Get strategy performance
        strategy_report = self.strategy_optimizer.get_performance_report()

        # Get calibration stats
        calibration_stats = self.confidence_estimator.get_calibration_stats()

        # Get meta-reflection if enough data
        if len(self.reflector.reflection_history) >= 3:
            meta_reflection = self.reflector.meta_reflect(
                self.reflector.reflection_history[-5:]
            )
        else:
            meta_reflection = "Insufficient data for meta-reflection"

        return f"""
METACOGNITION SYSTEM SELF-ASSESSMENT
=====================================

STRATEGY PERFORMANCE:
{json.dumps(strategy_report, indent=2)}

CONFIDENCE CALIBRATION:
{json.dumps(calibration_stats, indent=2)}

META-REFLECTION:
{meta_reflection}
"""


# =============================================================================
# Demonstration Functions
# =============================================================================

def demo_self_reflection():
    """Demonstrate self-reflection (without API)."""
    print("=" * 70)
    print("DEMO: Self-Reflection")
    print("=" * 70)

    print("""
Self-Reflection Process:

1. RECEIVE: Get a question and generated response
2. ANALYZE: Examine for accuracy, completeness, clarity
3. IDENTIFY: Find potential issues or weaknesses
4. ASSESS: Estimate confidence level
5. REVISE: Optionally generate improved response

Example Analysis:
- Question: "What causes inflation?"
- Response: "Inflation occurs when prices rise."

Reflection might identify:
- Issue: Response is too simplistic
- Issue: Doesn't explain WHY prices rise
- Confidence: 30% (incomplete)
- Suggestion: Add monetary policy, supply/demand factors
""")


def demo_confidence_estimation():
    """Demonstrate confidence estimation (without API)."""
    print("\n" + "=" * 70)
    print("DEMO: Confidence Estimation")
    print("=" * 70)

    print("""
Confidence Dimensions:

1. KNOWLEDGE CONFIDENCE
   - Is this topic well-understood?
   - Is the information up-to-date?
   - Are there known unknowns?

2. REASONING CONFIDENCE
   - Is the logic sound?
   - Are conclusions justified?
   - Are assumptions stated?

3. COMPLETENESS CONFIDENCE
   - Does it answer the full question?
   - Are edge cases covered?
   - Is context sufficient?

Calibration:
- Compare estimated confidence vs actual accuracy
- Adjust for systematic over/under-confidence
- Track performance over time
""")


def demo_complexity_scaling():
    """Demonstrate complexity scaling (without API)."""
    print("\n" + "=" * 70)
    print("DEMO: Adaptive Complexity Scaling")
    print("=" * 70)

    print("""
Complexity Levels:

SIMPLE (Level 1):
  "Machine learning is when computers learn from examples."

MODERATE (Level 2):
  "Machine learning uses algorithms to find patterns in data
   and make predictions without explicit programming."

COMPLEX (Level 3):
  "Machine learning encompasses supervised, unsupervised, and
   reinforcement learning paradigms, each with specific use cases..."

EXPERT (Level 5):
  "Consider the bias-variance tradeoff in model selection.
   Regularization techniques like L1/L2 penalties address overfitting..."

Adaptation triggers:
- User asks for clarification → Simplify
- User asks technical follow-up → Increase complexity
- User expertise indicators in language
""")


def demo_strategy_optimization():
    """Demonstrate strategy optimization (without API)."""
    print("\n" + "=" * 70)
    print("DEMO: Strategy Optimization")
    print("=" * 70)

    print("""
Strategy Performance Tracking:

Strategy         | Usage | Success Rate | Avg Confidence
-----------------+-------+--------------+---------------
direct           |   45  |     72%      |     65%
chain_of_thought |   32  |     85%      |     78%
few_shot         |   18  |     89%      |     82%
role_based       |   12  |     67%      |     58%

Optimization Insights:
- chain_of_thought works best for complex reasoning
- few_shot excels with pattern-based tasks
- role_based needs refinement for technical queries

Recommendation Engine:
- Matches context to historically successful strategies
- Learns from feedback to improve recommendations
""")


def demo_with_api():
    """
    Demonstrate metacognition with API calls.

    Requires a running vLLM server.
    """
    print("\n" + "=" * 70)
    print("DEMO: Metacognition System with API")
    print("=" * 70)

    system = MetacognitionSystem()

    # Test question and response
    question = "What are the main causes of climate change?"
    response = "Climate change is caused by greenhouse gases."

    print(f"\nQuestion: {question}")
    print(f"Initial Response: {response}")

    # Process with metacognition
    result = system.process_with_metacognition(question, response)

    print(f"\n--- Metacognitive Analysis ---")
    print(f"Complexity Level: {result['complexity_level']}")
    print(f"Confidence Scores:")
    for dim, score in result['confidence_assessment'].items():
        print(f"  {dim}: {score:.0%}")

    print(f"\nNeeds Caveats: {result['needs_caveats']}")
    print(f"Should Escalate: {result['should_escalate']}")

    print(f"\n--- Reflection ---")
    print(result['reflection'][:500] + "...")

    if result['revised_response']:
        print(f"\n--- Revised Response ---")
        print(result['revised_response'][:500] + "...")


def main():
    """Main entry point demonstrating the Metacognition Module."""
    print("\n" + "=" * 70)
    print("METACOGNITION MODULE - LLM Primitives")
    print("=" * 70)

    # Demos without API
    demo_self_reflection()
    demo_confidence_estimation()
    demo_complexity_scaling()
    demo_strategy_optimization()

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
