from graphraglet.LLM.LLM import LLM
from graphraglet.KnowledgeGraph import KnowledgeGraph

from typing import List, Tuple
from scipy.spatial.distance import cosine
import networkx as nx
import numpy as np

import logging

logger = logging.getLogger("GraphRAGlet")


def get_threshold(similarities: List[List[float]], sparsity: float) -> float:
    """
    Find a similiarty threshold that gives a desired sparsity of the knowledge graph

    Args:
        similarities (List[float]): Symmetric similarity matrix.
        sparsity (float): The desired sparsity.
    """

    # get an ordered list of similarities
    similarities = similarities[np.triu_indices(len(similarities), k=1)]
    similarities = similarities.flatten()
    # sort the similarities
    similarities = sorted(similarities)
    # get the threshold
    return similarities[int(len(similarities) * (1 - sparsity))]


def build_knowledge_graph(
    text_units: List[str], llm: LLM, sparsity: float = 0.2
) -> List[tuple[int, int]]:
    """
    Build the knowledge graph from the text units.

    Args:
        text_units (List[str]): The list of text units (indexed by order, deduplicated).
        llm (LLM): The LLM to use for embedding and similarity calculations.
        threshold (float, optional): The threshold for similarity. Defaults to 0.2.

    Returns:
        List[tuple]: The list of edges in the knowledge graph.
        List[List[float]]: The list of the embeddings for each text_unit
    """

    # get embeddings for each text unit
    embeddings = {text_unit: llm.get_embedding(text_unit) for text_unit in text_units}

    # create the similarity matrix (upper triangular)
    similarities = np.zeros((len(text_units), len(text_units)))
    for i in range(len(text_units)):
        for j in range(i + 1, len(text_units)):
            similarities[i][j] = cosine(
                embeddings[text_units[i]], embeddings[text_units[j]]
            )
            similarities[j][i] = similarities[i][j]

    # get the threshold
    threshold = get_threshold(similarities, sparsity)

    # build the knowledge graph
    knowledge_graph = []
    for i in range(len(text_units)):
        for j in range(i + 1, len(text_units)):
            if similarities[i][j] > threshold:
                knowledge_graph.append((i, j))

    return knowledge_graph, list(embeddings.values())


def community_detection(knowledge_graph: List[tuple]) -> List[List[int]]:
    """Perform community detection on the knowledge graph."""
    graph = nx.Graph()
    graph.add_edges_from(knowledge_graph)
    communities = nx.algorithms.community.louvain_communities(graph)
    return communities


def summarize_communities(
    communities: List[List[int]], text_units: List[str], llm: LLM
) -> Tuple[List[str], List[List[float]]]:
    """
    Summarize the communities in the knowledge graph.

    Returns:
        List[str]: summaries of the communities
        List[List[float]]: corresponding embeddings
    """

    # create summaries for each community
    summaries = []
    for community in communities:
        full_community_text = "\nFile 1: ".join([text_units[i] for i in community])
        summary = llm.summarize(full_community_text)
        summaries.append(summary)

    # embeddings of summaries
    comm_embeddings = [llm.get_embedding(com) for com in summaries]

    return summaries, comm_embeddings


def build_graphraglet(
    text_units: List[str], llm: LLM, threshold: float = 0.8
) -> List[str]:
    """Build the GraphRAGlet object."""
    logger.info("Building the knowledge graph...")
    text_unit_relations, text_unit_embeddings = build_knowledge_graph(
        text_units, llm, threshold
    )
    logger.info("Community detection...")
    communities = community_detection(text_unit_relations)
    logger.info("Summarizing communities...")
    summaries, summary_embeddings = summarize_communities(communities, text_units, llm)
    logger.info("Building GraphRAGlet object...")
    return KnowledgeGraph(
        text_units,
        text_unit_embeddings,
        summaries,
        summary_embeddings,
        communities,
        text_unit_relations,
    )
