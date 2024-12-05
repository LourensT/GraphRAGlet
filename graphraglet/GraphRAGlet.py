from typing import List
import networkx as nx

class GraphRAGlet:
    """GraphRAGlet class that contains the knowledge graph, summaries, and text units."""

    def __init__(self, text_units: List[str], community_summaries: List[str], communities: List[int], knowledge_graph: List[tuple]) -> None:
        """Initialize the GraphRAGlet class."""
        self.text_units = text_units
        self.community_summaries = community_summaries

        self.communities = communities
        self.graph = nx.Graph()
        self.graph.add_edges_from(knowledge_graph)
