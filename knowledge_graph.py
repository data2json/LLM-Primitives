#!/usr/bin/env python
"""
Knowledge Graph Module

This module demonstrates knowledge graph primitives for LLMs:
- Building and maintaining knowledge graphs
- Semantic network navigation and querying
- Hypothetical/speculative knowledge graphs
- Entity extraction and relationship mapping

Educational focus: Understanding how structured knowledge enhances LLM reasoning.
"""

import json
import requests
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Generator
from enum import Enum
import re


# =============================================================================
# Core Configuration
# =============================================================================

API_URL = "http://0.0.0.0:8000/v1/chat/completions"
DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


# =============================================================================
# Knowledge Graph Primitives
# =============================================================================

class NodeType(Enum):
    """Types of nodes in the knowledge graph."""
    ENTITY = "entity"           # Real-world entity
    CONCEPT = "concept"         # Abstract concept
    EVENT = "event"             # Temporal event
    HYPOTHESIS = "hypothesis"   # Speculative/hypothetical
    ATTRIBUTE = "attribute"     # Property or characteristic
    PROCESS = "process"         # Action or procedure


class EdgeType(Enum):
    """Types of relationships between nodes."""
    IS_A = "is_a"                  # Taxonomy/inheritance
    HAS_PROPERTY = "has_property"   # Attribute relationship
    PART_OF = "part_of"            # Composition
    CAUSES = "causes"              # Causal relationship
    RELATED_TO = "related_to"      # General association
    PRECEDES = "precedes"          # Temporal ordering
    CONTRADICTS = "contradicts"    # Logical contradiction
    SUPPORTS = "supports"          # Evidence relationship
    HYPOTHETICAL = "hypothetical"  # Speculative connection


class Certainty(Enum):
    """Certainty levels for knowledge."""
    FACT = 1.0          # Established fact
    HIGH = 0.8          # High confidence
    MEDIUM = 0.5        # Moderate confidence
    LOW = 0.3           # Low confidence
    SPECULATIVE = 0.1   # Purely hypothetical


@dataclass
class Node:
    """
    A node in the knowledge graph.

    Represents an entity, concept, or piece of knowledge.
    """
    id: str
    label: str
    node_type: NodeType = NodeType.ENTITY
    properties: dict = field(default_factory=dict)
    certainty: Certainty = Certainty.FACT
    source: str = "manual"
    created_at: datetime = field(default_factory=datetime.now)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.id == other.id
        return False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "type": self.node_type.value,
            "properties": self.properties,
            "certainty": self.certainty.value
        }


@dataclass
class Edge:
    """
    An edge (relationship) in the knowledge graph.

    Connects two nodes with a typed relationship.
    """
    source_id: str
    target_id: str
    edge_type: EdgeType = EdgeType.RELATED_TO
    weight: float = 1.0
    properties: dict = field(default_factory=dict)
    certainty: Certainty = Certainty.FACT
    bidirectional: bool = False

    def __hash__(self):
        return hash((self.source_id, self.target_id, self.edge_type))

    def to_dict(self) -> dict:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "type": self.edge_type.value,
            "weight": self.weight,
            "certainty": self.certainty.value
        }


# =============================================================================
# Knowledge Graph Implementation
# =============================================================================

class KnowledgeGraph:
    """
    A knowledge graph for storing and querying structured knowledge.

    Features:
    - Node and edge management
    - Graph traversal and querying
    - Subgraph extraction
    - Path finding
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self.nodes: dict = {}  # id -> Node
        self.edges: list = []
        self.adjacency: dict = defaultdict(list)  # source_id -> [(target_id, edge)]
        self.reverse_adjacency: dict = defaultdict(list)  # target_id -> [(source_id, edge)]

    def add_node(self, node: Node) -> Node:
        """Add a node to the graph."""
        self.nodes[node.id] = node
        return node

    def create_node(self, id: str, label: str, node_type: NodeType = NodeType.ENTITY,
                    certainty: Certainty = Certainty.FACT, **properties) -> Node:
        """Convenience method to create and add a node."""
        node = Node(
            id=id,
            label=label,
            node_type=node_type,
            certainty=certainty,
            properties=properties
        )
        return self.add_node(node)

    def add_edge(self, edge: Edge) -> Edge:
        """Add an edge to the graph."""
        self.edges.append(edge)
        self.adjacency[edge.source_id].append((edge.target_id, edge))
        self.reverse_adjacency[edge.target_id].append((edge.source_id, edge))

        if edge.bidirectional:
            self.adjacency[edge.target_id].append((edge.source_id, edge))
            self.reverse_adjacency[edge.source_id].append((edge.target_id, edge))

        return edge

    def create_edge(self, source_id: str, target_id: str,
                    edge_type: EdgeType = EdgeType.RELATED_TO,
                    certainty: Certainty = Certainty.FACT,
                    weight: float = 1.0, **properties) -> Edge:
        """Convenience method to create and add an edge."""
        edge = Edge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            certainty=certainty,
            weight=weight,
            properties=properties
        )
        return self.add_edge(edge)

    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def get_neighbors(self, node_id: str, edge_type: EdgeType = None) -> list:
        """Get all neighbors of a node, optionally filtered by edge type."""
        neighbors = []
        for target_id, edge in self.adjacency[node_id]:
            if edge_type is None or edge.edge_type == edge_type:
                if target_id in self.nodes:
                    neighbors.append((self.nodes[target_id], edge))
        return neighbors

    def get_incoming(self, node_id: str, edge_type: EdgeType = None) -> list:
        """Get all nodes that point to this node."""
        incoming = []
        for source_id, edge in self.reverse_adjacency[node_id]:
            if edge_type is None or edge.edge_type == edge_type:
                if source_id in self.nodes:
                    incoming.append((self.nodes[source_id], edge))
        return incoming

    def find_path(self, start_id: str, end_id: str, max_depth: int = 5) -> list:
        """Find a path between two nodes using BFS."""
        if start_id not in self.nodes or end_id not in self.nodes:
            return []

        visited = {start_id}
        queue = [(start_id, [start_id])]

        while queue:
            current_id, path = queue.pop(0)

            if len(path) > max_depth:
                continue

            if current_id == end_id:
                return path

            for neighbor_id, _ in self.adjacency[current_id]:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id]))

        return []

    def get_subgraph(self, center_id: str, depth: int = 2) -> "KnowledgeGraph":
        """Extract a subgraph centered on a node."""
        subgraph = KnowledgeGraph(f"{self.name}_subgraph")

        visited = set()
        queue = [(center_id, 0)]

        while queue:
            node_id, current_depth = queue.pop(0)

            if node_id in visited or current_depth > depth:
                continue

            visited.add(node_id)

            if node_id in self.nodes:
                subgraph.add_node(self.nodes[node_id])

            if current_depth < depth:
                for neighbor_id, edge in self.adjacency[node_id]:
                    if neighbor_id not in visited:
                        queue.append((neighbor_id, current_depth + 1))
                    if node_id in subgraph.nodes and neighbor_id in self.nodes:
                        subgraph.add_edge(edge)

        return subgraph

    def query_by_type(self, node_type: NodeType) -> list:
        """Get all nodes of a specific type."""
        return [n for n in self.nodes.values() if n.node_type == node_type]

    def query_by_certainty(self, min_certainty: float = 0.5) -> list:
        """Get all nodes above a certainty threshold."""
        return [n for n in self.nodes.values() if n.certainty.value >= min_certainty]

    def get_hypothetical_subgraph(self) -> "KnowledgeGraph":
        """Get all hypothetical/speculative knowledge."""
        subgraph = KnowledgeGraph(f"{self.name}_hypothetical")

        for node in self.nodes.values():
            if node.certainty == Certainty.SPECULATIVE or node.node_type == NodeType.HYPOTHESIS:
                subgraph.add_node(node)

        for edge in self.edges:
            if edge.source_id in subgraph.nodes and edge.target_id in subgraph.nodes:
                subgraph.add_edge(edge)

        return subgraph

    def to_dict(self) -> dict:
        """Serialize graph to dictionary."""
        return {
            "name": self.name,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges]
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize graph to JSON."""
        return json.dumps(self.to_dict(), indent=indent)

    def get_stats(self) -> dict:
        """Get statistics about the graph."""
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "node_types": {t.value: len(self.query_by_type(t)) for t in NodeType},
            "avg_degree": len(self.edges) * 2 / len(self.nodes) if self.nodes else 0
        }


# =============================================================================
# Knowledge Extraction (LLM-based)
# =============================================================================

class KnowledgeExtractor:
    """
    Extracts knowledge graph elements from text using an LLM.
    """

    def __init__(self, api_url: str = API_URL, model: str = DEFAULT_MODEL):
        self.api_url = api_url
        self.model = model

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

    def extract_entities(self, text: str) -> list:
        """Extract entities from text using LLM."""
        prompt = f"""Extract all named entities from the following text.
Return as a JSON array of objects with 'name', 'type' (person/organization/location/concept/event), and 'description'.

Text: {text}

Return ONLY valid JSON, no explanation:"""

        response = self._call_llm(prompt)

        try:
            # Try to extract JSON from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        return []

    def extract_relationships(self, text: str, entities: list = None) -> list:
        """Extract relationships between entities."""
        entity_list = ", ".join([e.get('name', str(e)) for e in (entities or [])])

        prompt = f"""Extract relationships between entities in this text.
Known entities: {entity_list}

Text: {text}

Return as a JSON array with 'source', 'target', 'relationship', and 'confidence' (0-1).
Return ONLY valid JSON:"""

        response = self._call_llm(prompt)

        try:
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        return []

    def build_graph_from_text(self, text: str, graph: KnowledgeGraph = None) -> KnowledgeGraph:
        """Build a knowledge graph from unstructured text."""
        graph = graph or KnowledgeGraph("extracted")

        # Extract entities
        entities = self.extract_entities(text)

        # Map entity types to node types
        type_map = {
            "person": NodeType.ENTITY,
            "organization": NodeType.ENTITY,
            "location": NodeType.ENTITY,
            "concept": NodeType.CONCEPT,
            "event": NodeType.EVENT,
        }

        # Add nodes
        for entity in entities:
            name = entity.get("name", "unknown")
            node_id = name.lower().replace(" ", "_")
            node_type = type_map.get(entity.get("type", "").lower(), NodeType.ENTITY)

            graph.create_node(
                id=node_id,
                label=name,
                node_type=node_type,
                description=entity.get("description", "")
            )

        # Extract and add relationships
        relationships = self.extract_relationships(text, entities)

        edge_type_map = {
            "is_a": EdgeType.IS_A,
            "part_of": EdgeType.PART_OF,
            "causes": EdgeType.CAUSES,
            "related": EdgeType.RELATED_TO,
        }

        for rel in relationships:
            source_id = rel.get("source", "").lower().replace(" ", "_")
            target_id = rel.get("target", "").lower().replace(" ", "_")

            if source_id in graph.nodes and target_id in graph.nodes:
                rel_type = rel.get("relationship", "related").lower()
                edge_type = edge_type_map.get(rel_type, EdgeType.RELATED_TO)
                confidence = float(rel.get("confidence", 0.5))

                certainty = Certainty.HIGH if confidence > 0.7 else (
                    Certainty.MEDIUM if confidence > 0.4 else Certainty.LOW
                )

                graph.create_edge(
                    source_id=source_id,
                    target_id=target_id,
                    edge_type=edge_type,
                    certainty=certainty,
                    weight=confidence
                )

        return graph


# =============================================================================
# Semantic Navigation
# =============================================================================

class SemanticNavigator:
    """
    Enables semantic navigation and reasoning over knowledge graphs.
    """

    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph

    def follow_chain(self, start_id: str, edge_types: list,
                     max_depth: int = 5) -> Generator:
        """
        Follow a chain of specific relationship types.

        Example: Follow IS_A -> HAS_PROPERTY chain
        """
        if not edge_types:
            return

        current_type_index = 0
        visited = set()

        def traverse(node_id: str, depth: int, type_idx: int):
            if depth > max_depth or node_id in visited:
                return

            visited.add(node_id)
            yield (node_id, self.graph.get_node(node_id), depth)

            current_edge_type = edge_types[type_idx % len(edge_types)]
            next_type_idx = (type_idx + 1) % len(edge_types)

            for neighbor, edge in self.graph.get_neighbors(node_id, current_edge_type):
                yield from traverse(neighbor.id, depth + 1, next_type_idx)

        yield from traverse(start_id, 0, 0)

    def find_common_ancestors(self, node_ids: list,
                              edge_type: EdgeType = EdgeType.IS_A) -> list:
        """Find common ancestors of multiple nodes."""
        if not node_ids:
            return []

        # Get ancestors for each node
        ancestor_sets = []
        for node_id in node_ids:
            ancestors = set()
            queue = [node_id]
            while queue:
                current = queue.pop(0)
                if current in ancestors:
                    continue
                ancestors.add(current)
                for neighbor, edge in self.graph.get_neighbors(current, edge_type):
                    queue.append(neighbor.id)
            ancestor_sets.append(ancestors)

        # Find intersection
        if ancestor_sets:
            common = ancestor_sets[0]
            for s in ancestor_sets[1:]:
                common = common.intersection(s)
            return list(common)

        return []

    def get_explanation_path(self, start_id: str, end_id: str) -> str:
        """Generate a natural language explanation of the path between nodes."""
        path = self.graph.find_path(start_id, end_id)

        if not path:
            return f"No connection found between {start_id} and {end_id}"

        explanations = []
        for i in range(len(path) - 1):
            source = self.graph.get_node(path[i])
            target = self.graph.get_node(path[i + 1])

            # Find the edge
            for neighbor_id, edge in self.graph.adjacency[path[i]]:
                if neighbor_id == path[i + 1]:
                    explanations.append(
                        f"{source.label} --[{edge.edge_type.value}]--> {target.label}"
                    )
                    break

        return " | ".join(explanations)

    def infer_relationships(self, node_id: str) -> list:
        """Infer potential new relationships based on graph patterns."""
        inferences = []
        node = self.graph.get_node(node_id)

        if not node:
            return inferences

        # Pattern: If A is_a B and B has_property C, then A might have C
        for parent, is_a_edge in self.graph.get_neighbors(node_id, EdgeType.IS_A):
            for prop, prop_edge in self.graph.get_neighbors(parent.id, EdgeType.HAS_PROPERTY):
                # Check if this property isn't already directly connected
                existing = [n.id for n, _ in self.graph.get_neighbors(node_id, EdgeType.HAS_PROPERTY)]
                if prop.id not in existing:
                    inferences.append({
                        "type": "inherited_property",
                        "source": node_id,
                        "target": prop.id,
                        "via": parent.id,
                        "confidence": 0.7,
                        "explanation": f"{node.label} might have {prop.label} because it is a {parent.label}"
                    })

        return inferences


# =============================================================================
# Hypothetical Knowledge Graphs
# =============================================================================

class HypotheticalGraphBuilder:
    """
    Builds speculative/hypothetical knowledge graphs.

    Useful for exploring "what-if" scenarios and counterfactuals.
    """

    def __init__(self, base_graph: KnowledgeGraph = None,
                 api_url: str = API_URL, model: str = DEFAULT_MODEL):
        self.base_graph = base_graph or KnowledgeGraph("hypothetical")
        self.api_url = api_url
        self.model = model

    def _call_llm(self, prompt: str) -> str:
        """Make an LLM API call."""
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
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

    def create_hypothesis_node(self, hypothesis: str,
                               related_to: list = None) -> Node:
        """Create a hypothetical node and connect it to existing nodes."""
        node_id = f"hyp_{len(self.base_graph.nodes)}"
        node = self.base_graph.create_node(
            id=node_id,
            label=hypothesis,
            node_type=NodeType.HYPOTHESIS,
            certainty=Certainty.SPECULATIVE
        )

        # Connect to related nodes
        for related_id in (related_to or []):
            if related_id in self.base_graph.nodes:
                self.base_graph.create_edge(
                    source_id=node_id,
                    target_id=related_id,
                    edge_type=EdgeType.HYPOTHETICAL,
                    certainty=Certainty.SPECULATIVE
                )

        return node

    def explore_counterfactual(self, premise: str, base_facts: list = None) -> dict:
        """
        Explore a counterfactual scenario using LLM.

        Args:
            premise: The "what if" premise
            base_facts: Known facts from the graph
        """
        facts_str = "\n".join([f"- {f}" for f in (base_facts or [])])

        prompt = f"""Given these known facts:
{facts_str}

Explore this counterfactual scenario: "{premise}"

Provide:
1. Immediate implications (direct consequences)
2. Secondary effects (ripple effects)
3. Potential contradictions with known facts
4. Confidence level (low/medium/high)

Format as a structured analysis:"""

        response = self._call_llm(prompt)

        return {
            "premise": premise,
            "analysis": response,
            "base_facts_used": len(base_facts or [])
        }

    def generate_hypotheses(self, context: str, num_hypotheses: int = 3) -> list:
        """Generate hypotheses based on context using LLM."""
        prompt = f"""Based on this context, generate {num_hypotheses} testable hypotheses:

Context: {context}

For each hypothesis, provide:
1. The hypothesis statement
2. What evidence would support it
3. What evidence would refute it
4. Initial confidence (0-1)

Return as JSON array with 'hypothesis', 'supporting_evidence', 'refuting_evidence', 'confidence':"""

        response = self._call_llm(prompt)

        try:
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                hypotheses = json.loads(json_match.group())
                return hypotheses
        except json.JSONDecodeError:
            pass

        return []


# =============================================================================
# Demonstration Functions
# =============================================================================

def demo_basic_graph():
    """Demonstrate basic knowledge graph operations."""
    print("=" * 70)
    print("DEMO: Basic Knowledge Graph Operations")
    print("=" * 70)

    graph = KnowledgeGraph("programming_concepts")

    # Create nodes
    graph.create_node("python", "Python", NodeType.CONCEPT)
    graph.create_node("programming_language", "Programming Language", NodeType.CONCEPT)
    graph.create_node("dynamic_typing", "Dynamic Typing", NodeType.ATTRIBUTE)
    graph.create_node("interpreted", "Interpreted", NodeType.ATTRIBUTE)
    graph.create_node("web_dev", "Web Development", NodeType.CONCEPT)
    graph.create_node("data_science", "Data Science", NodeType.CONCEPT)
    graph.create_node("machine_learning", "Machine Learning", NodeType.CONCEPT)

    # Create relationships
    graph.create_edge("python", "programming_language", EdgeType.IS_A)
    graph.create_edge("python", "dynamic_typing", EdgeType.HAS_PROPERTY)
    graph.create_edge("python", "interpreted", EdgeType.HAS_PROPERTY)
    graph.create_edge("python", "web_dev", EdgeType.RELATED_TO)
    graph.create_edge("python", "data_science", EdgeType.RELATED_TO)
    graph.create_edge("data_science", "machine_learning", EdgeType.RELATED_TO)
    graph.create_edge("machine_learning", "data_science", EdgeType.PART_OF)

    print("\n--- Graph Statistics ---")
    for key, value in graph.get_stats().items():
        print(f"{key}: {value}")

    print("\n--- Neighbors of 'python' ---")
    for neighbor, edge in graph.get_neighbors("python"):
        print(f"  -> {neighbor.label} ({edge.edge_type.value})")

    print("\n--- Path from 'python' to 'machine_learning' ---")
    path = graph.find_path("python", "machine_learning")
    print(" -> ".join(path))

    return graph


def demo_semantic_navigation(graph: KnowledgeGraph):
    """Demonstrate semantic navigation."""
    print("\n" + "=" * 70)
    print("DEMO: Semantic Navigation")
    print("=" * 70)

    navigator = SemanticNavigator(graph)

    print("\n--- Following IS_A -> RELATED_TO chain from 'python' ---")
    for node_id, node, depth in navigator.follow_chain(
        "python", [EdgeType.IS_A, EdgeType.RELATED_TO], max_depth=3
    ):
        indent = "  " * depth
        print(f"{indent}{node.label if node else node_id}")

    print("\n--- Explanation Path: python -> machine_learning ---")
    explanation = navigator.get_explanation_path("python", "machine_learning")
    print(explanation)

    print("\n--- Inferred Relationships for 'python' ---")
    inferences = navigator.infer_relationships("python")
    for inf in inferences:
        print(f"  {inf['explanation']}")


def demo_hypothetical_graph():
    """Demonstrate hypothetical/speculative knowledge graphs."""
    print("\n" + "=" * 70)
    print("DEMO: Hypothetical Knowledge Graphs")
    print("=" * 70)

    builder = HypotheticalGraphBuilder()

    # Create some base facts
    builder.base_graph.create_node("quantum_computing", "Quantum Computing", NodeType.CONCEPT)
    builder.base_graph.create_node("cryptography", "Cryptography", NodeType.CONCEPT)
    builder.base_graph.create_edge("quantum_computing", "cryptography", EdgeType.RELATED_TO)

    # Add a hypothesis
    hyp_node = builder.create_hypothesis_node(
        "Quantum computers will break RSA encryption by 2030",
        related_to=["quantum_computing", "cryptography"]
    )

    print(f"\nCreated hypothesis node: {hyp_node.label}")
    print(f"  Type: {hyp_node.node_type.value}")
    print(f"  Certainty: {hyp_node.certainty.value}")

    # Show the graph
    print("\n--- Hypothetical Subgraph ---")
    hyp_subgraph = builder.base_graph.get_hypothetical_subgraph()
    for node in hyp_subgraph.nodes.values():
        print(f"  {node.label} ({node.node_type.value})")


def demo_with_api():
    """
    Demonstrate LLM-based knowledge extraction.

    Requires a running vLLM server.
    """
    print("\n" + "=" * 70)
    print("DEMO: LLM-Based Knowledge Extraction")
    print("=" * 70)

    text = """
    Python was created by Guido van Rossum and first released in 1991.
    It emphasizes code readability with significant indentation.
    Python is widely used in web development, data science, and artificial intelligence.
    Major companies like Google, Netflix, and Instagram use Python extensively.
    """

    extractor = KnowledgeExtractor()
    graph = extractor.build_graph_from_text(text)

    print(f"\n--- Extracted Graph ---")
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Edges: {len(graph.edges)}")

    print("\nExtracted Nodes:")
    for node in graph.nodes.values():
        print(f"  - {node.label} ({node.node_type.value})")

    print("\nExtracted Relationships:")
    for edge in graph.edges:
        source = graph.get_node(edge.source_id)
        target = graph.get_node(edge.target_id)
        print(f"  - {source.label if source else edge.source_id} "
              f"--[{edge.edge_type.value}]--> "
              f"{target.label if target else edge.target_id}")


def main():
    """Main entry point demonstrating the Knowledge Graph Module."""
    print("\n" + "=" * 70)
    print("KNOWLEDGE GRAPH MODULE - LLM Primitives")
    print("=" * 70)

    # Demos without API
    graph = demo_basic_graph()
    demo_semantic_navigation(graph)
    demo_hypothetical_graph()

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
