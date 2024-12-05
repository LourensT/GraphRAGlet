# %%
from test_llms import test_llms
from test_build import test_build_graphraglet
from test_search import test_global_search, test_local_search, test_drift_search

def main():

    test_llms()

    data_path = "data/dulce/dulce.txt"
    raglet = test_build_graphraglet(data_path) 

    test_global_search(raglet)
    test_local_search(raglet)
    test_drift_search(raglet)

if __name__ == "__main__":
    main()

