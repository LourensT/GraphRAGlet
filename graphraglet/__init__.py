"""The GraphRAGlet package."""

from graphraglet.KnowledgeGraph import KnowledgeGraph
from graphraglet.build_graphraglet import build_graphraglet
from graphraglet.query_graphraglet import global_search, local_search, drift_search
from graphraglet.LLM.OAI import OAI

__all__ = [
    "KnowledgeGraph",
    "build_graphraglet",
    "global_search",
    "local_search",
    "drift_search",
    "OAI",
]


import logging
import datetime

# Now, create the logger for your package/module
logger = logging.getLogger("GraphRAGlet")

logger.setLevel(logging.DEBUG)  # Set the default logging level

# Create a file handler to log to a file with DEBUG level
file_handler = logging.FileHandler(f"GraphRAGlet_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
file_handler.setLevel(logging.DEBUG)  # Log all levels to the file

# Create a console handler to log to the console with INFO level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Log only INFO and above to the console

# Define the same format for both handlers
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# propogate the logger to the root logger
logger.propagate = True

# Example log messages
logger.debug("This is a debug message")
logger.info("This is an info message")