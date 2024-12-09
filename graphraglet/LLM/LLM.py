from typing import List
from scipy.spatial.distance import cosine


class LLM:
    """Interface for LLMs."""

    def get_embedding(self, text: str) -> List[float]:
        """Get the embedding of a text."""
        pass

    def get_similarity(self, text1: str, text2: str) -> float:
        """Get the similarity between two texts."""
        embedding1 = self.get_embedding(text1)
        embedding2 = self.get_embedding(text2)
        return cosine(embedding1, embedding2)

    def prompt(self, text: str) -> str:
        """Prompt the LLM to generate a response."""
        pass

    def summarize(self, text: str) -> str:
        """Summarize the text."""
        pass
