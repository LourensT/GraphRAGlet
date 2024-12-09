from sys import path

path.append("..")

from graphraglet import KnowledgeGraph, global_search, local_search, drift_search


def test_global_search(raglet: KnowledgeGraph, data_path: str):
    result = global_search(raglet, data_path)
    assert isinstance(result, list)


def test_local_search(raglet: KnowledgeGraph, data_path: str):
    result = local_search(raglet, data_path)
    assert isinstance(result, list)


def test_drift_search(raglet: KnowledgeGraph, data_path: str):
    result = drift_search(raglet, data_path)
    assert isinstance(result, list)
