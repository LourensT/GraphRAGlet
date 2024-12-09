# %%
from sys import path

path.append("..")

from graphraglet import build_graphraglet, OAI

from typing import List


def get_text_units(data_path: str) -> List[str]:
    """Get the text units from the data directory."""
    text_units = set()
    with open(data_path, "r") as f:
        for line in f:
            cleaned = line.strip()
            if cleaned != "" and cleaned not in text_units:
                text_units.add(cleaned)

    if data_path == "../data/dulce/dulce.txt":
        return clean_dulce_data(text_units)

    return list(text_units)


def clean_dulce_data(text_units: List[str]) -> List[str]:
    """Load the Dulce data."""
    # remove comments
    text_units = [t for t in text_units if not t.startswith("#")]
    return text_units


def test_build_graphraglet(data_path: str):
    text_units = get_text_units(data_path)
    llm = OAI()
    sparsity = 0.3

    raglet = build_graphraglet(text_units, llm, sparsity)
    return raglet


# %%
if __name__ == "__main__":
    # raglet = test_build_graphraglet("../data/play/play.txt")
    raglet = test_build_graphraglet("../data/dulce/dulce.txt")
    # %%
    print(raglet.text_units)
    print(raglet.community_summaries)
    print(raglet.communities)
    print(raglet.graph.nodes, raglet.graph.edges)

    raglet.visualize()

    # %% pickle the raglet
    # import pickle
    # with open("raglet_dulce_0.8.pickle", "wb") as f:
    #     pickle.dump(raglet, f)

