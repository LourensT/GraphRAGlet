from typing import List
import networkx as nx

class KnowledgeGraph:
    """Class that contains the knowledge graph, summaries, and text units."""

    def __init__(self, text_units: List[str], community_summaries: List[str], communities: List[int], knowledge_graph: List[tuple]) -> None:
        """Initialize the GraphRAGlet class."""
        self.text_units = text_units
        self.community_summaries = community_summaries

        self.communities = communities
        self.n_communities = len(communities)
        self.graph = nx.Graph()
        self.graph.add_edges_from(knowledge_graph)

    def visualize(self):
        """
        Show interactive visualization of the GraphRAGlet object.
        """
        import matplotlib.pyplot as plt

        # create list of colors of length equal to the number of communities
        colors = [plt.cm.tab20(i / self.n_communities) for i in range(self.n_communities)]
        
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
