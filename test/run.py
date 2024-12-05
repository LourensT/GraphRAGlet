from graphraglet import build_graphraglet, global_search, local_search, drift_search
from graphraglet.GraphRAGlet import GraphRAGlet

from typing import List

def main():
    data_path = "data/test_data.txt"
    raglet = test_build_graphraglet(data_path)
    test_global_search(raglet)
    test_local_search(raglet)
    test_drift_search(raglet)

def get_text_units(data_path: str) -> List[str]:
    """Get the text units from the data directory."""
    text_units = []
    with open(data_path, "r") as f:
        for line in f:
            text_units.append(line.strip())

    return text_units

def test_build_graphraglet(data_path: str):
    
    text_units = get_text_units(data_path)
    llm = None
    threshold = 0.8

    raglet = build_graphraglet(text_units, llm, threshold)
    assert isinstance(raglet, GraphRAGlet)
    return raglet

def test_global_search(raglet: GraphRAGlet, data_path: str):
    pass

def test_local_search(raglet: GraphRAGlet, data_path: str):
    pass

def test_drift_search(raglet: GraphRAGlet, data_path: str):
    pass

if __name__ == "__main__":
    main()
