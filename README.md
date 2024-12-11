# GraphRAGlet - Work in Progress

A light-weight "most of the way there" adaptation of the [GraphRAG](https://microsoft.github.io/graphrag/) package.

## What makes GraphRAGlet different from GraphRAG?
The goal of GraphRAGlet is to be a RAG search engine that is cheaper to index but has similar strenghts as GraphRAG.
In regular GraphRAG, indexing (building the knowledge graph) is expensive. GraphRAGlet aims to 
GraphRAGlet addresses this by building the knowledge graph through embeddings instead of extracting entities, relations and facts with an LLM. 
This is cheaper.

However, GraphRAG aims for a "global understanding" of the corpus, so that the original corpus is not needed in favor for the knowledge graph's community summaries. This is not the case for GraphRAGlet, which aims to return the most relevant documents for a query.

## How does GraphRAGlet work?
1. Make Text Units from Corpus
2. Generate semantic embeddings for text-units and build the knowledge graph through a cutoff threshold.
3. Community Detection on the  knowledge graph (using Leiden Algorithm)
4. Generate summaries of the communities

Then, with this augmented knowledge graph, we can implement the following search strategies:
* Global Search:
    1. Embedding Search on the community summaries
    2. RAG on the most relevant community
* Local Search: find the most relevant documents for a query within a specific community.

## Dev todo 
- [x] Set up logging
- [ ] Better chunking of text units
- [ ] Implement Search
    - [x] Global Search
        - [ ] parameterize
    - [ ] Local Search
    - [ ] Drift Search
- [ ] Set-up evaluations and testing
    - [ ] Multihop
- [ ] Add support for Groq or another LLaMa provider
- [ ] Use UMAP to reduce dimensionality of the embeddings for relation mining
- [ ] Parallelization 