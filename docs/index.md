# GFM-Hybrid

**Entity-Score-Guided Hybrid Retrieval with Iterative Chain-of-Thought Reasoning
for Multi-Hop and Medical Question Answering**

GFM-Hybrid is a **unified retrieval-and-reasoning pipeline** for multi-hop and
medical question answering. A **Graph Foundation Model (GFM-RAG)**, while reasoning
over a knowledge graph, produces an **entity-relevance tensor**
$P_q \in [0,1]^{|\mathcal{V}|}$. Instead of discarding this tensor, GFM-Hybrid
**keeps it and uses it to steer an entity-augmented BM25 searcher**. Graph and
lexical evidence are merged in a **step-local pool**, a **cross-encoder** does
precision ranking, and an **IRCoT** loop with structured JSON output drives the
process across multiple hops.

> The graph supplies **structural breadth**, while entity-augmented lexical matching
> recovers the **surface-level detail** an incomplete graph misses — crucial for
> medical text and low-resource languages such as Vietnamese.

## Four components

1. **Graph-Foundation Retriever with Entity Scores** — returns ranked documents and
   the min–max-normalised tensor $\tilde{P}_q$.
   → `gfmrag_hybrid/gfm/retriever_with_entity_scores.py`
2. **Entity-Augmented BM25 Retrieval** — concatenates seed entities + high-scoring
   graph entities + the sub-question into one BM25 query.
   → `gfmrag_hybrid/bm25/searcher.py` (`BM25Searcher`)
3. **Step-Local Global Chunk Pool** — merges both branches, refreshed every step.
   → `gfmrag_hybrid/workflow/core_engine.py`
4. **Cross-Encoder Reranking + IRCoT** — `BAAI/bge-reranker-v2-m3` plus an LLM
   emitting a 5-field JSON object.
   → `gfmrag_hybrid/workflow/core_engine.py` (`agent_reasoning_with_reranker`)

## Get started

- [Installation](install.md)
- [Data Preparation](workflow/data_preparation.md)
- [KG-index Construction](workflow/kg_index.md)
- [Retrieval](workflow/inference.md)
- [Training](workflow/training.md)

GFM-Hybrid extends **GFM-RAG** (Luo et al., NeurIPS 2025) and draws inspiration from
IRCoT, HippoRAG, GraphRAG, LightRAG and the BGE reranker/embedding family.