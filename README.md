# GraphRAGlet

A minimal adaptation of the GraphRAG package.

## How does GraphRAGlet work?
1. Make Text Units from Corpus
2. Extract Entities, Relations, Facts from Text Units and build the knowledge graph with LLMs
3. Community Detection on the  knowledge graph (using Leiden Algorithm)
4. Generate summaries of the communities

Then, with this augmented knowledge graph, we can implement different query strategies:
1. Global Search: find the most relevant documents for a query.
2. Local Search: find the most relevant documents for a query within a specific community.

## What makes GraphRAGlet different from GraphRAG?

GraphRAGlet is a minimal adaptation of the GraphRAG package. An expensive step in the process is the building of the knowledge graph. 
GraphRAGlet addresses this by building the knowledge graph through embeddings instead of extracting entities, relations and facts with an LLM. 
This is orders of magnitude faster and more efficient.


## Dev todo 
- [x] Set up logging
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