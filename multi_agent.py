#!/usr/bin/env python
"""
Multi-Agent Simulation Module

This module demonstrates multi-agent primitives for LLMs:
- Agent creation with distinct personas and capabilities
- Inter-agent communication and collaboration
- Debate and consensus-building patterns
- Emergent behavior through agent interactions
- Dynamic persona adaptation

Educational focus: Understanding how multiple AI agents can collaborate.
"""

import json
import requests
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Callable
from enum import Enum
import random


# =============================================================================
# Core Configuration
# =============================================================================

API_URL = "http://0.0.0.0:8000/v1/chat/completions"
DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


# =============================================================================
# Agent Types and Structures
# =============================================================================

class AgentRole(Enum):
    """Predefined agent roles."""
    ANALYST = "analyst"           # Analyzes information
    CRITIC = "critic"             # Provides critical feedback
    CREATIVE = "creative"         # Generates creative solutions
    PLANNER = "planner"           # Creates plans and strategies
    EXECUTOR = "executor"         # Focuses on implementation
    MEDIATOR = "mediator"         # Facilitates consensus
    EXPERT = "expert"             # Domain specialist
    DEVIL_ADVOCATE = "devil_advocate"  # Challenges assumptions


class CommunicationStyle(Enum):
    """Communication styles for agents."""
    FORMAL = "formal"
    CASUAL = "casual"
    TECHNICAL = "technical"
    SOCRATIC = "socratic"
    ASSERTIVE = "assertive"
    COLLABORATIVE = "collaborative"


@dataclass
class AgentPersona:
    """
    Defines an agent's personality and behavior patterns.
    """
    name: str
    role: AgentRole
    expertise: list = field(default_factory=list)
    communication_style: CommunicationStyle = CommunicationStyle.FORMAL
    personality_traits: list = field(default_factory=list)
    system_prompt: str = ""

    def __post_init__(self):
        if not self.system_prompt:
            self.system_prompt = self._generate_system_prompt()

    def _generate_system_prompt(self) -> str:
        """Generate a system prompt from persona attributes."""
        traits = ", ".join(self.personality_traits) if self.personality_traits else "helpful and thorough"
        expertise = ", ".join(self.expertise) if self.expertise else "general knowledge"

        return f"""You are {self.name}, a {self.role.value} with expertise in {expertise}.

Your communication style is {self.communication_style.value}.
Your personality traits: {traits}

When responding:
- Stay in character as {self.name}
- Apply your expertise to analyze problems
- Communicate in your designated style
- Express your unique perspective based on your role"""


@dataclass
class Message:
    """A message in agent communication."""
    sender: str
    recipient: str  # Can be "all" for broadcast
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    message_type: str = "statement"  # statement, question, response, proposal
    metadata: dict = field(default_factory=dict)


@dataclass
class ConversationLog:
    """Log of all messages in a multi-agent conversation."""
    messages: list = field(default_factory=list)

    def add(self, message: Message):
        self.messages.append(message)

    def get_history_for_agent(self, agent_name: str, limit: int = 10) -> list:
        """Get relevant messages for an agent."""
        relevant = [
            m for m in self.messages
            if m.recipient in (agent_name, "all") or m.sender == agent_name
        ]
        return relevant[-limit:]

    def to_string(self, limit: int = None) -> str:
        """Convert log to readable string."""
        msgs = self.messages[-limit:] if limit else self.messages
        return "\n".join([
            f"[{m.sender} -> {m.recipient}]: {m.content}"
            for m in msgs
        ])


# =============================================================================
# Agent Implementation
# =============================================================================

class Agent:
    """
    An AI agent with a distinct persona that can communicate and collaborate.
    """

    def __init__(self, persona: AgentPersona,
                 api_url: str = API_URL, model: str = DEFAULT_MODEL):
        self.persona = persona
        self.api_url = api_url
        self.model = model
        self.memory: list = []  # Agent's personal memory
        self.state: dict = {}   # Agent's current state

    @property
    def name(self) -> str:
        return self.persona.name

    def _call_llm(self, messages: list, temperature: float = 0.7) -> str:
        """Make an LLM API call."""
        data = {
            "model": self.model,
            "messages": messages,
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

    def think(self, context: str) -> str:
        """Internal reasoning before responding."""
        messages = [
            {"role": "system", "content": self.persona.system_prompt},
            {"role": "user", "content": f"""As {self.name}, consider this situation:

{context}

What are your thoughts? Think through this step by step from your unique perspective."""}
        ]

        return self._call_llm(messages, temperature=0.5)

    def respond(self, message: Message, conversation_log: ConversationLog) -> Message:
        """Generate a response to a message."""
        # Build context from conversation history
        history = conversation_log.get_history_for_agent(self.name)
        history_str = "\n".join([
            f"{m.sender}: {m.content}" for m in history
        ])

        messages = [
            {"role": "system", "content": self.persona.system_prompt},
            {"role": "user", "content": f"""Conversation so far:
{history_str}

Latest message from {message.sender}:
{message.content}

Respond as {self.name}. Stay in character and provide your perspective:"""}
        ]

        response_content = self._call_llm(messages)

        return Message(
            sender=self.name,
            recipient=message.sender,
            content=response_content,
            message_type="response"
        )

    def propose(self, topic: str, conversation_log: ConversationLog) -> Message:
        """Propose an idea or solution."""
        history_str = conversation_log.to_string(limit=5)

        messages = [
            {"role": "system", "content": self.persona.system_prompt},
            {"role": "user", "content": f"""Context from discussion:
{history_str}

Topic: {topic}

As {self.name}, propose a solution or idea. Be specific and actionable:"""}
        ]

        response_content = self._call_llm(messages)

        return Message(
            sender=self.name,
            recipient="all",
            content=response_content,
            message_type="proposal"
        )

    def critique(self, proposal: Message, conversation_log: ConversationLog) -> Message:
        """Provide critical feedback on a proposal."""
        messages = [
            {"role": "system", "content": self.persona.system_prompt},
            {"role": "user", "content": f"""A proposal was made by {proposal.sender}:

"{proposal.content}"

As {self.name}, provide constructive critique:
1. What are the strengths?
2. What are the weaknesses?
3. What improvements would you suggest?
4. What risks should be considered?"""}
        ]

        response_content = self._call_llm(messages)

        return Message(
            sender=self.name,
            recipient=proposal.sender,
            content=response_content,
            message_type="critique"
        )


# =============================================================================
# Predefined Personas
# =============================================================================

PREDEFINED_PERSONAS = {
    "technical_lead": AgentPersona(
        name="Alex",
        role=AgentRole.EXPERT,
        expertise=["software architecture", "system design", "best practices"],
        communication_style=CommunicationStyle.TECHNICAL,
        personality_traits=["detail-oriented", "pragmatic", "experienced"]
    ),

    "product_manager": AgentPersona(
        name="Jordan",
        role=AgentRole.PLANNER,
        expertise=["product strategy", "user needs", "market analysis"],
        communication_style=CommunicationStyle.COLLABORATIVE,
        personality_traits=["user-focused", "strategic", "communicative"]
    ),

    "creative_designer": AgentPersona(
        name="Sam",
        role=AgentRole.CREATIVE,
        expertise=["UX design", "innovation", "user experience"],
        communication_style=CommunicationStyle.CASUAL,
        personality_traits=["creative", "empathetic", "visual thinker"]
    ),

    "quality_analyst": AgentPersona(
        name="Morgan",
        role=AgentRole.CRITIC,
        expertise=["testing", "quality assurance", "risk assessment"],
        communication_style=CommunicationStyle.FORMAL,
        personality_traits=["thorough", "skeptical", "detail-oriented"]
    ),

    "devil_advocate": AgentPersona(
        name="Casey",
        role=AgentRole.DEVIL_ADVOCATE,
        expertise=["critical thinking", "risk identification", "alternative viewpoints"],
        communication_style=CommunicationStyle.SOCRATIC,
        personality_traits=["contrarian", "analytical", "provocative"]
    ),

    "mediator": AgentPersona(
        name="Riley",
        role=AgentRole.MEDIATOR,
        expertise=["facilitation", "conflict resolution", "consensus building"],
        communication_style=CommunicationStyle.COLLABORATIVE,
        personality_traits=["diplomatic", "patient", "neutral"]
    ),
}


# =============================================================================
# Multi-Agent Orchestration Patterns
# =============================================================================

class MultiAgentOrchestrator:
    """
    Orchestrates interactions between multiple agents.
    """

    def __init__(self, api_url: str = API_URL, model: str = DEFAULT_MODEL):
        self.api_url = api_url
        self.model = model
        self.agents: dict = {}  # name -> Agent
        self.conversation_log = ConversationLog()

    def add_agent(self, persona: AgentPersona) -> Agent:
        """Add an agent to the simulation."""
        agent = Agent(persona, self.api_url, self.model)
        self.agents[agent.name] = agent
        return agent

    def add_predefined_agent(self, persona_key: str) -> Agent:
        """Add a predefined agent."""
        if persona_key not in PREDEFINED_PERSONAS:
            raise ValueError(f"Unknown persona: {persona_key}")
        return self.add_agent(PREDEFINED_PERSONAS[persona_key])

    def broadcast(self, sender_name: str, content: str) -> Message:
        """Send a message to all agents."""
        message = Message(
            sender=sender_name,
            recipient="all",
            content=content,
            message_type="statement"
        )
        self.conversation_log.add(message)
        return message

    def run_round_robin(self, topic: str, rounds: int = 2) -> list:
        """
        Run a round-robin discussion where each agent speaks in turn.
        """
        results = []

        # Initial topic broadcast
        self.broadcast("Facilitator", f"Discussion topic: {topic}")

        for round_num in range(rounds):
            for agent in self.agents.values():
                # Each agent proposes or responds
                if round_num == 0:
                    message = agent.propose(topic, self.conversation_log)
                else:
                    # Get last message from another agent
                    last_msg = self.conversation_log.messages[-1]
                    message = agent.respond(last_msg, self.conversation_log)

                self.conversation_log.add(message)
                results.append({
                    "round": round_num + 1,
                    "agent": agent.name,
                    "message": message.content
                })

        return results

    def run_debate(self, proposition: str, pro_agent: str, con_agent: str,
                   rounds: int = 2) -> dict:
        """
        Run a structured debate between two agents.
        """
        if pro_agent not in self.agents or con_agent not in self.agents:
            raise ValueError("Agents not found")

        pro = self.agents[pro_agent]
        con = self.agents[con_agent]

        debate_log = []

        # Opening statements
        self.broadcast("Moderator", f"Debate topic: {proposition}")

        for round_num in range(rounds):
            # Pro argument
            pro_msg = pro.propose(
                f"{'Argue FOR' if round_num == 0 else 'Rebut and argue FOR'}: {proposition}",
                self.conversation_log
            )
            self.conversation_log.add(pro_msg)
            debate_log.append({"round": round_num + 1, "side": "pro", "content": pro_msg.content})

            # Con argument
            con_msg = con.propose(
                f"{'Argue AGAINST' if round_num == 0 else 'Rebut and argue AGAINST'}: {proposition}",
                self.conversation_log
            )
            self.conversation_log.add(con_msg)
            debate_log.append({"round": round_num + 1, "side": "con", "content": con_msg.content})

        return {
            "proposition": proposition,
            "pro_agent": pro_agent,
            "con_agent": con_agent,
            "debate": debate_log
        }

    def run_critique_session(self, proposal: str, proposer: str,
                            critics: list) -> dict:
        """
        Run a critique session where multiple agents critique a proposal.
        """
        if proposer not in self.agents:
            raise ValueError(f"Proposer '{proposer}' not found")

        results = {
            "proposal": proposal,
            "proposer": proposer,
            "critiques": []
        }

        # Proposer makes the proposal
        proposal_msg = Message(
            sender=proposer,
            recipient="all",
            content=proposal,
            message_type="proposal"
        )
        self.conversation_log.add(proposal_msg)

        # Each critic provides feedback
        for critic_name in critics:
            if critic_name in self.agents:
                critic = self.agents[critic_name]
                critique_msg = critic.critique(proposal_msg, self.conversation_log)
                self.conversation_log.add(critique_msg)
                results["critiques"].append({
                    "critic": critic_name,
                    "feedback": critique_msg.content
                })

        return results

    def build_consensus(self, topic: str, max_rounds: int = 3) -> dict:
        """
        Facilitate a consensus-building discussion.
        """
        results = {
            "topic": topic,
            "rounds": [],
            "final_consensus": None
        }

        # Check if mediator exists
        mediator = None
        for agent in self.agents.values():
            if agent.persona.role == AgentRole.MEDIATOR:
                mediator = agent
                break

        self.broadcast("Facilitator", f"Let's build consensus on: {topic}")

        for round_num in range(max_rounds):
            round_results = {"round": round_num + 1, "contributions": []}

            # Each agent contributes
            for agent in self.agents.values():
                if agent.persona.role != AgentRole.MEDIATOR:
                    msg = agent.propose(
                        f"Contribute to consensus on: {topic}",
                        self.conversation_log
                    )
                    self.conversation_log.add(msg)
                    round_results["contributions"].append({
                        "agent": agent.name,
                        "contribution": msg.content
                    })

            # Mediator synthesizes (if present)
            if mediator:
                synthesis_msg = mediator.propose(
                    f"Synthesize the contributions and identify common ground on: {topic}",
                    self.conversation_log
                )
                self.conversation_log.add(synthesis_msg)
                round_results["synthesis"] = synthesis_msg.content

            results["rounds"].append(round_results)

        # Final consensus from mediator
        if mediator:
            final_msg = mediator.propose(
                f"Provide the final consensus statement on: {topic}",
                self.conversation_log
            )
            results["final_consensus"] = final_msg.content

        return results


# =============================================================================
# Dynamic Persona Adaptation
# =============================================================================

class AdaptiveAgent(Agent):
    """
    An agent that can adapt its persona based on user interactions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptation_history = []
        self.user_profile = {}

    def analyze_user(self, user_messages: list) -> dict:
        """Analyze user characteristics from their messages."""
        messages_str = "\n".join(user_messages[-5:])

        messages = [
            {"role": "system", "content": "You are an expert at understanding communication styles."},
            {"role": "user", "content": f"""Analyze these messages from a user:

{messages_str}

Determine:
1. Technical expertise level (beginner/intermediate/expert)
2. Preferred communication style (formal/casual/technical)
3. Key interests or concerns
4. Apparent mood or urgency

Provide analysis as JSON:"""}
        ]

        response = self._call_llm(messages, temperature=0.3)

        # Try to parse JSON from response
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass

        return {
            "expertise": "intermediate",
            "style": "formal",
            "interests": [],
            "mood": "neutral"
        }

    def adapt_persona(self, user_profile: dict):
        """Adapt persona based on user profile."""
        self.user_profile = user_profile

        # Adapt communication style
        style_map = {
            "beginner": CommunicationStyle.CASUAL,
            "intermediate": CommunicationStyle.COLLABORATIVE,
            "expert": CommunicationStyle.TECHNICAL
        }

        expertise = user_profile.get("expertise", "intermediate")
        self.persona.communication_style = style_map.get(expertise, CommunicationStyle.FORMAL)

        # Update system prompt
        self.persona.system_prompt = f"""{self.persona._generate_system_prompt()}

ADAPTATION NOTES:
- User expertise level: {expertise}
- User prefers: {user_profile.get('style', 'formal')} communication
- Key concerns: {', '.join(user_profile.get('interests', []))}
- Adapt your responses accordingly."""

        self.adaptation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user_profile": user_profile
        })


# =============================================================================
# Emergent Behavior Simulation
# =============================================================================

class EmergentBehaviorSimulator:
    """
    Simulates emergent behavior through agent interactions.
    """

    def __init__(self, api_url: str = API_URL, model: str = DEFAULT_MODEL):
        self.orchestrator = MultiAgentOrchestrator(api_url, model)
        self.interaction_count = 0
        self.emergent_patterns = []

    def setup_ecosystem(self, agent_configs: list):
        """Set up an ecosystem of agents."""
        for config in agent_configs:
            if isinstance(config, str):
                self.orchestrator.add_predefined_agent(config)
            elif isinstance(config, AgentPersona):
                self.orchestrator.add_agent(config)

    def run_free_interaction(self, seed_topic: str, steps: int = 5) -> list:
        """
        Run free-form interactions where agents respond to each other.

        Emergent behavior can arise from these unstructured interactions.
        """
        results = []

        # Seed the conversation
        self.orchestrator.broadcast("Observer", f"Initial stimulus: {seed_topic}")

        agents_list = list(self.orchestrator.agents.values())

        for step in range(steps):
            # Randomly select responding agent
            responding_agent = random.choice(agents_list)

            # Get recent messages
            recent = self.orchestrator.conversation_log.messages[-3:]
            if not recent:
                continue

            # Agent responds to the most recent relevant message
            trigger = recent[-1]
            response = responding_agent.respond(trigger, self.orchestrator.conversation_log)
            self.orchestrator.conversation_log.add(response)

            results.append({
                "step": step + 1,
                "agent": responding_agent.name,
                "triggered_by": trigger.sender,
                "response": response.content[:200] + "..."
            })

            self.interaction_count += 1

        return results

    def detect_patterns(self) -> list:
        """Analyze conversation for emergent patterns."""
        # Simple pattern detection based on message content overlap
        messages = self.orchestrator.conversation_log.messages

        patterns = []

        # Detect recurring themes
        all_content = " ".join([m.content.lower() for m in messages])
        words = all_content.split()
        word_freq = {}
        for word in words:
            if len(word) > 5:  # Only longer words
                word_freq[word] = word_freq.get(word, 0) + 1

        # Top recurring terms
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        if sorted_words:
            patterns.append({
                "type": "recurring_themes",
                "themes": [w for w, c in sorted_words[:5] if c > 1]
            })

        # Detect agent interaction patterns
        interaction_pairs = {}
        for m in messages:
            if m.recipient != "all":
                pair = tuple(sorted([m.sender, m.recipient]))
                interaction_pairs[pair] = interaction_pairs.get(pair, 0) + 1

        if interaction_pairs:
            most_interactive = max(interaction_pairs.items(), key=lambda x: x[1])
            patterns.append({
                "type": "strong_interaction_pair",
                "agents": most_interactive[0],
                "interaction_count": most_interactive[1]
            })

        self.emergent_patterns = patterns
        return patterns


# =============================================================================
# Demonstration Functions
# =============================================================================

def demo_agent_personas():
    """Demonstrate agent persona creation (without API)."""
    print("=" * 70)
    print("DEMO: Agent Personas")
    print("=" * 70)

    print("\nPredefined Personas Available:")
    for key, persona in PREDEFINED_PERSONAS.items():
        print(f"\n--- {key} ---")
        print(f"Name: {persona.name}")
        print(f"Role: {persona.role.value}")
        print(f"Expertise: {', '.join(persona.expertise)}")
        print(f"Style: {persona.communication_style.value}")


def demo_orchestration_patterns():
    """Demonstrate orchestration patterns (without API)."""
    print("\n" + "=" * 70)
    print("DEMO: Multi-Agent Orchestration Patterns")
    print("=" * 70)

    print("""
Available Patterns:

1. ROUND ROBIN
   - Each agent speaks in turn
   - Good for gathering diverse perspectives
   - Ensures all voices are heard

2. DEBATE
   - Two agents argue opposing sides
   - Structured with opening statements and rebuttals
   - Good for exploring pros and cons

3. CRITIQUE SESSION
   - One agent proposes, others critique
   - Good for stress-testing ideas
   - Identifies weaknesses and risks

4. CONSENSUS BUILDING
   - All agents contribute iteratively
   - Mediator synthesizes common ground
   - Good for reaching agreement
""")


def demo_with_api():
    """
    Demonstrate multi-agent simulation with API calls.

    Requires a running vLLM server.
    """
    print("\n" + "=" * 70)
    print("DEMO: Multi-Agent Simulation with API")
    print("=" * 70)

    orchestrator = MultiAgentOrchestrator()

    # Add agents
    orchestrator.add_predefined_agent("technical_lead")
    orchestrator.add_predefined_agent("product_manager")
    orchestrator.add_predefined_agent("devil_advocate")

    print("\nAgents added:")
    for name in orchestrator.agents:
        print(f"  - {name}")

    # Run a round robin discussion
    print("\n--- Round Robin Discussion ---")
    topic = "Should we adopt microservices architecture?"

    results = orchestrator.run_round_robin(topic, rounds=1)

    for r in results:
        print(f"\n[Round {r['round']}] {r['agent']}:")
        print(r['message'][:300] + "...")

    # Run critique session
    print("\n--- Critique Session ---")
    proposal = "We should implement a complete rewrite of the legacy system using modern frameworks."

    critique_results = orchestrator.run_critique_session(
        proposal=proposal,
        proposer="Alex",
        critics=["Jordan", "Casey"]
    )

    for critique in critique_results["critiques"]:
        print(f"\n{critique['critic']}'s Feedback:")
        print(critique['feedback'][:300] + "...")


def demo_emergent_behavior():
    """
    Demonstrate emergent behavior simulation.

    Requires a running vLLM server.
    """
    print("\n" + "=" * 70)
    print("DEMO: Emergent Behavior Simulation")
    print("=" * 70)

    simulator = EmergentBehaviorSimulator()

    # Set up ecosystem
    simulator.setup_ecosystem([
        "technical_lead",
        "creative_designer",
        "quality_analyst"
    ])

    print("\nRunning free interaction simulation...")
    results = simulator.run_free_interaction(
        "What makes a great user experience?",
        steps=3
    )

    for r in results:
        print(f"\nStep {r['step']}: {r['agent']} (triggered by {r['triggered_by']})")
        print(r['response'])

    # Detect patterns
    patterns = simulator.detect_patterns()
    print("\n--- Emergent Patterns Detected ---")
    for p in patterns:
        print(f"Type: {p['type']}")
        print(f"Details: {p}")


def main():
    """Main entry point demonstrating the Multi-Agent Simulation Module."""
    print("\n" + "=" * 70)
    print("MULTI-AGENT SIMULATION MODULE - LLM Primitives")
    print("=" * 70)

    # Demos without API
    demo_agent_personas()
    demo_orchestration_patterns()

    # Demo with API
    print("\n" + "=" * 70)
    print("API DEMO (requires running vLLM server)")
    print("=" * 70)

    try:
        demo_with_api()
        demo_emergent_behavior()
    except Exception as e:
        print(f"\nNote: API demos skipped - {e}")
        print("Start a vLLM server to run API-dependent demos.")


if __name__ == "__main__":
    main()
