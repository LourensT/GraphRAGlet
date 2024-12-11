# %%
from test_llms import test_llms
from test_build import test_build_graphraglet
from test_search import test_global_search

def main():

    test_llms()

    # data_path = "data/dulce/dulce.txt"
    data_path = "../data/play/play.txt"
    raglet = test_build_graphraglet(data_path) 

    test_global_search(raglet)


if __name__ == "__main__":
    main()
