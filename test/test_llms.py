from sys import path
path.append("..")

from graphraglet.LLM.LLM import LLM
from graphraglet.LLM.OAI import OAI

def test_llms():
    test_one(OAI())
    print("LLM test passed for OAI.")

def test_one(implementation: LLM, verbose: bool = False):
    some_text = "This is some text."
    embedding = implementation.get_embedding(some_text)

    some_different_text = "This is some different text."
    embedding_different = implementation.get_embedding(some_different_text)
    assert embedding != embedding_different

    comsimilarity = implementation.get_similarity(some_text, some_different_text)
    if verbose:
        print(f"Similarity is {comsimilarity}")

    instruction = "Complete the following text: " + some_text
    completion = implementation.prompt(instruction)
    assert isinstance(completion, str)
    if verbose:
        print(f"Completion is {completion}")

    longer_text = "I say blah. I say blah blah. I say blah blah blah."
    summary = implementation.summarize(longer_text)
    if verbose:
        print(f"Summary is {summary}")

if __name__ == "__main__":
    test_llms()
    print("All LLM test passed.")