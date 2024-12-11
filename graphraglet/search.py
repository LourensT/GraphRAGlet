from typing import List

from graphraglet.KnowledgeGraph import KnowledgeGraph
from graphraglet.LLM.LLM import LLM

import logging

logger = logging.getLogger("GraphRAGlet")


def global_search(raglet: KnowledgeGraph, llm: LLM, query: str) -> List[str]:
    """Perform a global search on the Knowledge_graph object."""

    query_embedding = llm.get_embedding(query)

    # 1. RAG on community summaries
    top_communities_index = raglet.top_k_communities(query_embedding, 1)[1][0]

    # 2. Detect which communities to inspect further
    docs = raglet.top_k_in_community(query_embedding, top_communities_index, 3)

    return docs


def local_search(graphraglet: KnowledgeGraph, query: str, community: int) -> List[str]:
    """Perform a local search of a community on the GraphRAGlet object."""
    pass


def drift_search(graphraglet: KnowledgeGraph, query: str, text_unit: int) -> List[str]:
    """Perform a drift search of a community on the GraphRAGlet object."""
    pass
