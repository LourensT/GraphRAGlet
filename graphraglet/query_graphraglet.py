from typing import List

from graphraglet.GraphRAGlet import GraphRAGlet

import logging
logger = logging.getLogger("GraphRAGlet")

def global_search(graphraglet: GraphRAGlet, query: str) -> List[str]:
    """Perform a global search on the GraphRAGlet object."""
    pass

def local_search(graphraglet: GraphRAGlet, query: str, community: int) -> List[str]:
    """Perform a local search of a community on the GraphRAGlet object."""
    pass

def drift_search(graphraglet: GraphRAGlet, query: str, text_unit: int) -> List[str]:
    """Perform a drift search of a community on the GraphRAGlet object."""
    pass
