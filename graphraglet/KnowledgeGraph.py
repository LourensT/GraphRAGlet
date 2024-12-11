from typing import List, Tuple
import networkx as nx
from scipy.spatial.distance import cosine
import numpy as np

import logging

logger = logging.getLogger("GraphRAGlet")


class KnowledgeGraph:
    """Class that contains the knowledge graph, summaries, and text units."""

    def __init__(
        self,
        text_units: List[str],
        text_unit_embeddings: List[List[float]],
        community_summaries: List[str],
        community_summary_embeddings: List[List[float]],
        communities: List[int],
        relations: List[tuple],
    ) -> None:
        """Initialize the GraphRAGlet class."""

        self.text_units = text_units
        self.text_unit_embeddings = text_unit_embeddings

        self.communities = communities
        self.n_communities = len(communities)
        self.community_summaries = community_summaries
        self.community_embeddings = community_summary_embeddings

        self.graph = nx.Graph()
        self.graph.add_edges_from(relations)

        logger.info("Built knowledge graph.")

    def top_k_communities(
        self, query_embedding: List[float], k: int
    ) -> Tuple[List[str], List[int]]:
        """
        Return the top $k$ most relevant communities

        Args:
            List[str]: community summaries
            List[int]: community indices
        """
        dists = []
        for embed in self.community_embeddings:
            dists.append(cosine(embed, query_embedding))

        # sort index
        indices = np.argsort(dists)

        return [self.community_summaries[i] for i in indices[:k]], indices

    def top_k_in_community(
        self, query_embedding: List[float], community_index: int, k: int
    ) -> List[str]:
        """
        Return the top $k$ test units within a given community

        """
        community_nodes = self.communities[community_index]
        rel_text_units = [self.text_units[i] for i in community_nodes]
        rel_embeddings = [self.text_unit_embeddings[i] for i in community_nodes]

        dists = []
        for embed in rel_embeddings:
            dists.append(cosine(embed, query_embedding))

        # sort index
        indices = np.argsort(dists)

        return [rel_text_units[i] for i in indices[:k]]

    def visualize(self):
        """
        Show interactive visualization of the GraphRAGlet object.
        """
        logger.info("Visualizing the knowledge graph...")

        import matplotlib.pyplot as plt

        # create list of colors of length equal to the number of communities
        colors = [
            plt.cm.tab20(i / self.n_communities) for i in range(self.n_communities)
        ]

        # create a color list for each node
        node_colors = []
        for i in range(len(self.text_units)):
            for j in range(self.n_communities):
                if i in self.communities[j]:
                    node_colors.append(colors[j])

        # positions
        pos = nx.spring_layout(self.graph)

        # Add labels
        node_labels = {}
        for node, label in zip(self.graph.nodes, self.text_units):
            node_labels[node] = label[:20] + "..."

        nx.draw_networkx_labels(self.graph, pos, labels=node_labels, font_size=5)

        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, node_size=1)

        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, edge_color="black", width=0.1)

        plt.show()
