from sys import path

path.append("..")

from graphraglet import KnowledgeGraph, global_search, local_search, drift_search, OAI


def test_global_search(raglet: KnowledgeGraph):
    query = "This is some user query"
    oai = OAI()

    result = global_search(raglet, oai, query)
    print(result)
    assert isinstance(result, list)


def test_local_search(raglet: KnowledgeGraph, data_path: str):
    pass


def test_drift_search(raglet: KnowledgeGraph, data_path: str):
    pass
