#!/usr/bin/env python
"""
Reasoning Engine Module

This module demonstrates reasoning primitives for LLMs:
- Analogical reasoning (finding and applying analogies)
- Counterfactual reasoning (exploring "what if" scenarios)
- Temporal reasoning (past, present, future chains)
- Causal reasoning (cause and effect analysis)
- Recursive problem decomposition

Educational focus: Understanding different reasoning patterns and their applications.
"""

import json
import requests
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from enum import Enum


# =============================================================================
# Core Configuration
# =============================================================================

API_URL = "http://0.0.0.0:8000/v1/chat/completions"
DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


# =============================================================================
# Reasoning Types and Structures
# =============================================================================

class ReasoningType(Enum):
    """Types of reasoning supported by the engine."""
    DEDUCTIVE = "deductive"         # General to specific
    INDUCTIVE = "inductive"         # Specific to general
    ABDUCTIVE = "abductive"         # Best explanation
    ANALOGICAL = "analogical"       # Similarity-based
    COUNTERFACTUAL = "counterfactual"  # What-if
    CAUSAL = "causal"               # Cause-effect
    TEMPORAL = "temporal"           # Time-based
    RECURSIVE = "recursive"         # Decomposition


@dataclass
class ReasoningStep:
    """A single step in a reasoning chain."""
    step_number: int
    reasoning_type: ReasoningType
    premise: str
    conclusion: str
    confidence: float = 1.0
    evidence: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "step": self.step_number,
            "type": self.reasoning_type.value,
            "premise": self.premise,
            "conclusion": self.conclusion,
            "confidence": self.confidence,
            "evidence": self.evidence
        }


@dataclass
class ReasoningChain:
    """A complete chain of reasoning steps."""
    name: str
    goal: str
    steps: list = field(default_factory=list)
    final_conclusion: str = ""
    overall_confidence: float = 1.0

    def add_step(self, step: ReasoningStep):
        self.steps.append(step)
        # Update overall confidence as product of step confidences
        self.overall_confidence = 1.0
        for s in self.steps:
            self.overall_confidence *= s.confidence

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "final_conclusion": self.final_conclusion,
            "overall_confidence": self.overall_confidence
        }


# =============================================================================
# Abstract Reasoning Primitive
# =============================================================================

class ReasoningPrimitive(ABC):
    """Abstract base class for reasoning primitives."""

    def __init__(self, api_url: str = API_URL, model: str = DEFAULT_MODEL):
        self.api_url = api_url
        self.model = model

    def _call_llm(self, prompt: str, temperature: float = 0.3,
                  max_tokens: int = 1024) -> str:
        """Make an LLM API call."""
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
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
        except Exception as e:
            return f"[Error: {e}]"

    @abstractmethod
    def reason(self, input_data: dict) -> dict:
        """Apply this reasoning primitive to input data."""
        pass


# =============================================================================
# Analogical Reasoning
# =============================================================================

@dataclass
class Analogy:
    """Represents an analogy between source and target domains."""
    source_domain: str
    target_domain: str
    mappings: dict  # source_element -> target_element
    strength: float = 0.5
    explanation: str = ""


class AnalogicalReasoning(ReasoningPrimitive):
    """
    Implements analogical reasoning.

    Finds structural similarities between domains and transfers knowledge.
    """

    def find_analogy(self, source: str, target: str) -> Analogy:
        """Find analogies between source and target concepts."""
        prompt = f"""Analyze the structural similarity between these two domains:

Source domain: {source}
Target domain: {target}

Identify:
1. Key elements in each domain
2. Structural mappings between elements
3. How knowledge from source applies to target
4. Strength of the analogy (0-1)
5. Limitations of this analogy

Provide a detailed analysis:"""

        response = self._call_llm(prompt)

        return Analogy(
            source_domain=source,
            target_domain=target,
            mappings={},  # Would be extracted from response
            strength=0.5,
            explanation=response
        )

    def transfer_solution(self, source_problem: str, source_solution: str,
                         target_problem: str) -> dict:
        """Transfer a solution from source domain to target domain."""
        prompt = f"""Using analogical reasoning, transfer this solution approach:

SOURCE PROBLEM: {source_problem}
SOURCE SOLUTION: {source_solution}

TARGET PROBLEM: {target_problem}

Apply the solution approach from the source to the target:
1. Identify key principles from the source solution
2. Map these principles to the target domain
3. Adapt the solution for the target context
4. Note any limitations or adjustments needed

Transferred solution:"""

        response = self._call_llm(prompt)

        return {
            "source_problem": source_problem,
            "source_solution": source_solution,
            "target_problem": target_problem,
            "transferred_solution": response,
            "reasoning_type": ReasoningType.ANALOGICAL.value
        }

    def reason(self, input_data: dict) -> dict:
        """Apply analogical reasoning."""
        if "source_solution" in input_data:
            return self.transfer_solution(
                input_data.get("source_problem", ""),
                input_data.get("source_solution", ""),
                input_data.get("target_problem", "")
            )
        else:
            analogy = self.find_analogy(
                input_data.get("source", ""),
                input_data.get("target", "")
            )
            return {
                "analogy": analogy.explanation,
                "strength": analogy.strength
            }


# =============================================================================
# Counterfactual Reasoning
# =============================================================================

@dataclass
class Counterfactual:
    """Represents a counterfactual scenario."""
    original_fact: str
    counterfactual_premise: str
    immediate_effects: list = field(default_factory=list)
    downstream_effects: list = field(default_factory=list)
    contradictions: list = field(default_factory=list)


class CounterfactualReasoning(ReasoningPrimitive):
    """
    Implements counterfactual reasoning.

    Explores "what if" scenarios and their implications.
    """

    def explore_counterfactual(self, fact: str, counterfactual: str,
                               context: str = "") -> Counterfactual:
        """Explore the implications of a counterfactual scenario."""
        prompt = f"""Explore this counterfactual scenario:

ORIGINAL FACT: {fact}
COUNTERFACTUAL: {counterfactual}
{f"CONTEXT: {context}" if context else ""}

Analyze:
1. IMMEDIATE EFFECTS: Direct consequences if the counterfactual were true
2. DOWNSTREAM EFFECTS: Secondary and tertiary consequences
3. CONTRADICTIONS: What known facts would be contradicted
4. PLAUSIBILITY: How plausible is this counterfactual world

Provide a thorough counterfactual analysis:"""

        response = self._call_llm(prompt)

        return Counterfactual(
            original_fact=fact,
            counterfactual_premise=counterfactual,
            immediate_effects=[response]  # Would be parsed
        )

    def compare_scenarios(self, scenario_a: str, scenario_b: str,
                          criteria: list = None) -> dict:
        """Compare two counterfactual scenarios."""
        criteria_str = ", ".join(criteria) if criteria else "feasibility, impact, risks"

        prompt = f"""Compare these two scenarios:

SCENARIO A: {scenario_a}
SCENARIO B: {scenario_b}

Evaluation criteria: {criteria_str}

For each scenario, analyze:
1. Likelihood of occurrence
2. Positive outcomes
3. Negative outcomes
4. Overall assessment

Then provide a comparative recommendation:"""

        response = self._call_llm(prompt)

        return {
            "scenario_a": scenario_a,
            "scenario_b": scenario_b,
            "comparison": response,
            "reasoning_type": ReasoningType.COUNTERFACTUAL.value
        }

    def reason(self, input_data: dict) -> dict:
        """Apply counterfactual reasoning."""
        if "scenario_b" in input_data:
            return self.compare_scenarios(
                input_data.get("scenario_a", ""),
                input_data.get("scenario_b", ""),
                input_data.get("criteria", [])
            )
        else:
            cf = self.explore_counterfactual(
                input_data.get("fact", ""),
                input_data.get("counterfactual", ""),
                input_data.get("context", "")
            )
            return {
                "original": cf.original_fact,
                "counterfactual": cf.counterfactual_premise,
                "analysis": cf.immediate_effects[0] if cf.immediate_effects else ""
            }


# =============================================================================
# Temporal Reasoning
# =============================================================================

class TemporalFrame(Enum):
    """Temporal frames for reasoning."""
    PAST = "past"
    PRESENT = "present"
    FUTURE = "future"


@dataclass
class TemporalChain:
    """A chain of temporally connected events/states."""
    events: list = field(default_factory=list)  # [(timestamp_or_order, event)]
    causal_links: list = field(default_factory=list)


class TemporalReasoning(ReasoningPrimitive):
    """
    Implements temporal reasoning.

    Reasons about past, present, and future states and transitions.
    """

    def analyze_temporal_sequence(self, events: list) -> dict:
        """Analyze a sequence of events temporally."""
        events_str = "\n".join([f"{i+1}. {e}" for i, e in enumerate(events)])

        prompt = f"""Analyze this sequence of events:

{events_str}

Provide:
1. CHRONOLOGICAL ANALYSIS: Order and timing relationships
2. CAUSAL CHAINS: Which events caused or influenced others
3. PATTERNS: Recurring patterns or cycles
4. PREDICTIONS: What might happen next based on the pattern
5. HISTORICAL CONTEXT: How past events inform current state

Temporal analysis:"""

        response = self._call_llm(prompt)

        return {
            "events": events,
            "analysis": response,
            "reasoning_type": ReasoningType.TEMPORAL.value
        }

    def project_future(self, current_state: str, trends: list,
                       time_horizon: str = "near-term") -> dict:
        """Project future states based on current state and trends."""
        trends_str = "\n".join([f"- {t}" for t in trends])

        prompt = f"""Project the future based on current conditions:

CURRENT STATE: {current_state}

OBSERVED TRENDS:
{trends_str}

TIME HORIZON: {time_horizon}

Provide:
1. MOST LIKELY SCENARIO: Expected outcome if trends continue
2. BEST CASE: Optimistic scenario
3. WORST CASE: Pessimistic scenario
4. KEY INFLECTION POINTS: Critical moments that could change trajectory
5. CONFIDENCE LEVEL: How certain is this projection

Future projection:"""

        response = self._call_llm(prompt)

        return {
            "current_state": current_state,
            "trends": trends,
            "time_horizon": time_horizon,
            "projection": response,
            "reasoning_type": ReasoningType.TEMPORAL.value
        }

    def trace_causation(self, effect: str, context: str = "") -> dict:
        """Trace back the causal chain leading to an effect."""
        prompt = f"""Trace the causal chain leading to this effect:

EFFECT: {effect}
{f"CONTEXT: {context}" if context else ""}

Work backwards to identify:
1. IMMEDIATE CAUSES: Direct factors that led to this effect
2. ROOT CAUSES: Fundamental underlying causes
3. CONTRIBUTING FACTORS: Secondary influences
4. CAUSAL CHAIN: The sequence of cause-effect relationships
5. PREVENTABILITY: Could this have been prevented? How?

Causal analysis:"""

        response = self._call_llm(prompt)

        return {
            "effect": effect,
            "causal_analysis": response,
            "reasoning_type": ReasoningType.TEMPORAL.value
        }

    def reason(self, input_data: dict) -> dict:
        """Apply temporal reasoning."""
        if "events" in input_data:
            return self.analyze_temporal_sequence(input_data["events"])
        elif "trends" in input_data:
            return self.project_future(
                input_data.get("current_state", ""),
                input_data.get("trends", []),
                input_data.get("time_horizon", "near-term")
            )
        else:
            return self.trace_causation(
                input_data.get("effect", ""),
                input_data.get("context", "")
            )


# =============================================================================
# Causal Reasoning
# =============================================================================

class CausalReasoning(ReasoningPrimitive):
    """
    Implements causal reasoning.

    Analyzes cause-effect relationships and causal structures.
    """

    def identify_causes(self, effect: str, candidates: list = None) -> dict:
        """Identify potential causes of an observed effect."""
        candidates_str = "\n".join([f"- {c}" for c in candidates]) if candidates else "Unknown"

        prompt = f"""Identify the causes of this effect:

OBSERVED EFFECT: {effect}

CANDIDATE CAUSES:
{candidates_str}

Analyze:
1. DIRECT CAUSES: Factors directly producing the effect
2. INDIRECT CAUSES: Factors contributing through intermediaries
3. NECESSARY CONDITIONS: What must be true for the effect to occur
4. SUFFICIENT CONDITIONS: What alone could produce the effect
5. CAUSAL STRENGTH: How strong is each causal link (weak/moderate/strong)

Causal identification:"""

        response = self._call_llm(prompt)

        return {
            "effect": effect,
            "candidates": candidates,
            "causal_analysis": response,
            "reasoning_type": ReasoningType.CAUSAL.value
        }

    def predict_effects(self, cause: str, system_context: str = "") -> dict:
        """Predict effects of a given cause."""
        prompt = f"""Predict the effects of this cause:

CAUSE: {cause}
{f"SYSTEM CONTEXT: {system_context}" if system_context else ""}

Predict:
1. IMMEDIATE EFFECTS: Direct consequences
2. SECONDARY EFFECTS: Downstream consequences
3. SIDE EFFECTS: Unintended consequences
4. FEEDBACK LOOPS: Self-reinforcing or self-correcting dynamics
5. TIMELINE: When would effects manifest (immediate/short-term/long-term)

Effect prediction:"""

        response = self._call_llm(prompt)

        return {
            "cause": cause,
            "context": system_context,
            "predicted_effects": response,
            "reasoning_type": ReasoningType.CAUSAL.value
        }

    def analyze_intervention(self, system: str, intervention: str,
                            desired_outcome: str) -> dict:
        """Analyze the effectiveness of an intervention."""
        prompt = f"""Analyze this intervention:

SYSTEM: {system}
INTERVENTION: {intervention}
DESIRED OUTCOME: {desired_outcome}

Evaluate:
1. MECHANISM: How would this intervention work?
2. EFFECTIVENESS: Likelihood of achieving desired outcome (0-100%)
3. SIDE EFFECTS: Potential unintended consequences
4. ALTERNATIVES: Other interventions that might work
5. RECOMMENDATION: Should this intervention be implemented?

Intervention analysis:"""

        response = self._call_llm(prompt)

        return {
            "system": system,
            "intervention": intervention,
            "desired_outcome": desired_outcome,
            "analysis": response,
            "reasoning_type": ReasoningType.CAUSAL.value
        }

    def reason(self, input_data: dict) -> dict:
        """Apply causal reasoning."""
        if "intervention" in input_data:
            return self.analyze_intervention(
                input_data.get("system", ""),
                input_data["intervention"],
                input_data.get("desired_outcome", "")
            )
        elif "candidates" in input_data or "effect" in input_data:
            return self.identify_causes(
                input_data.get("effect", ""),
                input_data.get("candidates", [])
            )
        else:
            return self.predict_effects(
                input_data.get("cause", ""),
                input_data.get("context", "")
            )


# =============================================================================
# Recursive Problem Decomposition
# =============================================================================

@dataclass
class SubProblem:
    """A sub-problem in recursive decomposition."""
    id: str
    description: str
    parent_id: Optional[str] = None
    solution: Optional[str] = None
    status: str = "pending"  # pending, solved, blocked


class RecursiveDecomposition(ReasoningPrimitive):
    """
    Implements recursive problem decomposition.

    Breaks complex problems into sub-problems and reconstructs solutions.
    """

    def __init__(self, *args, max_depth: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_depth = max_depth
        self.sub_problems: dict = {}  # id -> SubProblem

    def decompose(self, problem: str, depth: int = 0) -> dict:
        """Decompose a problem into sub-problems."""
        if depth >= self.max_depth:
            return {"problem": problem, "sub_problems": [], "atomic": True}

        prompt = f"""Decompose this problem into smaller sub-problems:

PROBLEM: {problem}

Rules:
1. Break into 2-4 sub-problems
2. Each sub-problem should be simpler than the original
3. Sub-problems should be independent when possible
4. Together, solving all sub-problems should solve the original

Return as a numbered list of sub-problems with brief descriptions:"""

        response = self._call_llm(prompt)

        # Parse sub-problems from response
        sub_problems = []
        lines = response.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                # Remove number/bullet and clean
                clean = line.lstrip("0123456789.-) ").strip()
                if clean:
                    sub_problems.append(clean)

        return {
            "problem": problem,
            "sub_problems": sub_problems,
            "depth": depth,
            "atomic": len(sub_problems) == 0
        }

    def solve_atomic(self, problem: str) -> str:
        """Solve an atomic (non-decomposable) problem."""
        prompt = f"""Solve this specific problem directly:

PROBLEM: {problem}

Provide a clear, concise solution:"""

        return self._call_llm(prompt)

    def reconstruct_solution(self, problem: str, sub_solutions: list) -> str:
        """Reconstruct full solution from sub-solutions."""
        solutions_str = "\n\n".join([
            f"Sub-problem {i+1}: {s['problem']}\nSolution: {s['solution']}"
            for i, s in enumerate(sub_solutions)
        ])

        prompt = f"""Reconstruct a complete solution from these sub-solutions:

ORIGINAL PROBLEM: {problem}

SUB-SOLUTIONS:
{solutions_str}

Synthesize these into a coherent, complete solution to the original problem:"""

        return self._call_llm(prompt)

    def solve_recursively(self, problem: str, depth: int = 0) -> dict:
        """Recursively solve a problem through decomposition."""
        # Decompose
        decomposition = self.decompose(problem, depth)

        if decomposition["atomic"] or not decomposition["sub_problems"]:
            # Base case: solve directly
            solution = self.solve_atomic(problem)
            return {
                "problem": problem,
                "solution": solution,
                "method": "direct",
                "depth": depth
            }

        # Recursive case: solve sub-problems
        sub_solutions = []
        for sub_problem in decomposition["sub_problems"]:
            sub_result = self.solve_recursively(sub_problem, depth + 1)
            sub_solutions.append(sub_result)

        # Reconstruct
        final_solution = self.reconstruct_solution(problem, sub_solutions)

        return {
            "problem": problem,
            "solution": final_solution,
            "sub_solutions": sub_solutions,
            "method": "decomposition",
            "depth": depth
        }

    def reason(self, input_data: dict) -> dict:
        """Apply recursive decomposition."""
        problem = input_data.get("problem", "")
        if input_data.get("decompose_only", False):
            return self.decompose(problem)
        else:
            return self.solve_recursively(problem)


# =============================================================================
# Unified Reasoning Engine
# =============================================================================

class ReasoningEngine:
    """
    Unified reasoning engine that orchestrates multiple reasoning primitives.
    """

    def __init__(self, api_url: str = API_URL, model: str = DEFAULT_MODEL):
        self.api_url = api_url
        self.model = model

        # Initialize reasoning primitives
        self.primitives = {
            ReasoningType.ANALOGICAL: AnalogicalReasoning(api_url, model),
            ReasoningType.COUNTERFACTUAL: CounterfactualReasoning(api_url, model),
            ReasoningType.TEMPORAL: TemporalReasoning(api_url, model),
            ReasoningType.CAUSAL: CausalReasoning(api_url, model),
            ReasoningType.RECURSIVE: RecursiveDecomposition(api_url, model),
        }

        self.reasoning_history = []

    def reason(self, reasoning_type: ReasoningType, input_data: dict) -> dict:
        """Apply a specific reasoning primitive."""
        if reasoning_type not in self.primitives:
            return {"error": f"Unknown reasoning type: {reasoning_type}"}

        result = self.primitives[reasoning_type].reason(input_data)

        # Track history
        self.reasoning_history.append({
            "type": reasoning_type.value,
            "input": input_data,
            "output": result,
            "timestamp": datetime.now().isoformat()
        })

        return result

    def chain_reasoning(self, steps: list) -> ReasoningChain:
        """
        Execute a chain of reasoning steps.

        Args:
            steps: List of (reasoning_type, input_data) tuples
        """
        chain = ReasoningChain(
            name="chained_reasoning",
            goal="Multi-step reasoning"
        )

        accumulated_context = {}

        for i, (reasoning_type, input_data) in enumerate(steps):
            # Merge accumulated context
            merged_input = {**accumulated_context, **input_data}

            result = self.reason(reasoning_type, merged_input)

            step = ReasoningStep(
                step_number=i + 1,
                reasoning_type=reasoning_type,
                premise=str(input_data),
                conclusion=str(result.get("analysis", result.get("solution", str(result)))),
                confidence=0.8  # Would be computed from result
            )
            chain.add_step(step)

            # Update accumulated context with results
            accumulated_context.update(result)

        chain.final_conclusion = str(accumulated_context)
        return chain

    def select_reasoning_type(self, problem_description: str) -> ReasoningType:
        """
        Automatically select the best reasoning type for a problem.

        Uses heuristics based on problem keywords.
        """
        problem_lower = problem_description.lower()

        # Simple heuristic matching
        if any(word in problem_lower for word in ["like", "similar", "analogy", "compare to"]):
            return ReasoningType.ANALOGICAL
        elif any(word in problem_lower for word in ["what if", "hypothetical", "imagine", "suppose"]):
            return ReasoningType.COUNTERFACTUAL
        elif any(word in problem_lower for word in ["when", "before", "after", "timeline", "sequence"]):
            return ReasoningType.TEMPORAL
        elif any(word in problem_lower for word in ["cause", "effect", "why", "because", "lead to"]):
            return ReasoningType.CAUSAL
        elif any(word in problem_lower for word in ["complex", "break down", "steps", "parts"]):
            return ReasoningType.RECURSIVE
        else:
            return ReasoningType.CAUSAL  # Default


# =============================================================================
# Demonstration Functions
# =============================================================================

def demo_analogical_reasoning():
    """Demonstrate analogical reasoning (without API)."""
    print("=" * 70)
    print("DEMO: Analogical Reasoning")
    print("=" * 70)

    print("""
Analogical Reasoning transfers knowledge from a source domain to a target domain.

Example:
- Source: "How does the immune system fight viruses?"
- Target: "How should a computer system fight malware?"

The reasoner finds structural mappings:
- Virus → Malware
- White blood cells → Antivirus software
- Antibodies → Virus signatures
- Immune memory → Definition updates

This allows solutions from medicine to inform cybersecurity.
""")


def demo_counterfactual_reasoning():
    """Demonstrate counterfactual reasoning (without API)."""
    print("\n" + "=" * 70)
    print("DEMO: Counterfactual Reasoning")
    print("=" * 70)

    print("""
Counterfactual Reasoning explores "what if" scenarios.

Example:
- Fact: "The project was delayed because of poor communication"
- Counterfactual: "What if the team had daily standups?"

Analysis:
1. Immediate effects: Issues would surface faster
2. Downstream effects: Earlier problem resolution, better morale
3. Contradictions: Might conflict with async work preferences
4. Plausibility: High - many teams successfully use daily standups

This helps understand causal relationships and plan improvements.
""")


def demo_temporal_reasoning():
    """Demonstrate temporal reasoning (without API)."""
    print("\n" + "=" * 70)
    print("DEMO: Temporal Reasoning")
    print("=" * 70)

    print("""
Temporal Reasoning analyzes sequences and time-based patterns.

Example sequence:
1. Market research conducted (Month 1)
2. Product designed (Month 2-3)
3. MVP developed (Month 4-6)
4. Beta testing (Month 7)
5. Launch (Month 8)

Analysis:
- Causal chain: Research → Design → Development → Testing → Launch
- Pattern: Waterfall-like sequential progression
- Prediction: Post-launch support and iteration will follow
- Risk: Late discovery of issues due to sequential nature
""")


def demo_recursive_decomposition():
    """Demonstrate recursive decomposition (without API)."""
    print("\n" + "=" * 70)
    print("DEMO: Recursive Problem Decomposition")
    print("=" * 70)

    print("""
Recursive Decomposition breaks complex problems into simpler sub-problems.

Example: "Build a web application for task management"

Decomposition:
├── Backend Development
│   ├── Database design
│   ├── API endpoints
│   └── Authentication
├── Frontend Development
│   ├── UI components
│   ├── State management
│   └── API integration
├── DevOps
│   ├── CI/CD setup
│   ├── Hosting configuration
│   └── Monitoring

Each leaf node is solved directly, then solutions are reconstructed
upward to solve the original problem.
""")


def demo_with_api():
    """
    Demonstrate the reasoning engine with API calls.

    Requires a running vLLM server.
    """
    print("\n" + "=" * 70)
    print("DEMO: Reasoning Engine with API")
    print("=" * 70)

    engine = ReasoningEngine()

    # Test analogical reasoning
    print("\n--- Analogical Reasoning ---")
    result = engine.reason(
        ReasoningType.ANALOGICAL,
        {
            "source": "How a river system distributes water",
            "target": "How a content delivery network distributes data"
        }
    )
    print(result.get("analogy", str(result))[:500] + "...")

    # Test causal reasoning
    print("\n--- Causal Reasoning ---")
    result = engine.reason(
        ReasoningType.CAUSAL,
        {
            "cause": "Implementing automated testing",
            "context": "Software development project"
        }
    )
    print(result.get("predicted_effects", str(result))[:500] + "...")

    # Test recursive decomposition
    print("\n--- Recursive Decomposition ---")
    result = engine.reason(
        ReasoningType.RECURSIVE,
        {
            "problem": "Implement a user authentication system",
            "decompose_only": True
        }
    )
    print(f"Sub-problems identified: {len(result.get('sub_problems', []))}")
    for i, sp in enumerate(result.get("sub_problems", [])[:5]):
        print(f"  {i+1}. {sp}")


def main():
    """Main entry point demonstrating the Reasoning Engine Module."""
    print("\n" + "=" * 70)
    print("REASONING ENGINE MODULE - LLM Primitives")
    print("=" * 70)

    # Demos without API
    demo_analogical_reasoning()
    demo_counterfactual_reasoning()
    demo_temporal_reasoning()
    demo_recursive_decomposition()

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
