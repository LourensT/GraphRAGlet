from graphraglet.LLM.LLM import LLM

from openai import OpenAI

from typing import List
import os

import logging

logger = logging.getLogger("GraphRAGlet")


class OAI(LLM):
    """
    Interface for OpenAI's API.

    Make sure to set the OPENAI_API_KEY environment variable.
    """

    EMBEDDING_MODEL = "text-embedding-3-small"
    COMPLETION_MODEL = "gpt-4o"

    def __init__(self, openai_api_key: str = None) -> None:
        """Initialize the OpenAI class."""
        if "OPENAI_API_KEY" not in os.environ:
            if openai_api_key is None:
                raise ValueError("OPENAI_API_KEY environment variable not set.")
            else:
                os.environ["OPENAI_API_KEY"] = openai_api_key

        self.client = OpenAI()

        logger.info("Initialized OpenAI LLM class.")

    def get_embedding(self, text: str) -> List[float]:
        """Get the embedding of a text."""
        response = self.client.embeddings.create(
            input=[text], model="text-embedding-3-small"
        )

        return response.data[0].embedding

    def prompt(self, text: str) -> str:
        logger.info("Prompting the LLM...")
        logger.debug(f"Prompt: {text}")
        completion = self.client.chat.completions.create(
            model=self.COMPLETION_MODEL,
            messages=[{"role": "user", "content": text}],
            temperature=0.1,
        )

        logger.debug(f"Completion: {completion.choices[0].message.content}")

        return completion.choices[0].message.content

    def summarize(self, text: str) -> str:
        logger.info("Summarizing the text...")
        """Summarize the text."""
        full_prompt = f"Summarize the following text: {text}"
        return self.prompt(full_prompt)
