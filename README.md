# GFM-Hybrid

**Entity-Score-Guided Hybrid Retrieval with Iterative Chain-of-Thought Reasoning for Multi-Hop and Medical Question Answering**

This is the **source package (`gfmrag_hybrid`)** of GFM-Hybrid — a **unified
retrieval-and-reasoning pipeline** for multi-hop and medical question answering.
Core idea: a **Graph Foundation Model (GFM-RAG)**, while reasoning over a knowledge
graph, produces an **entity-relevance tensor** $P_q \in [0,1]^{|\mathcal{V}|}$.
Instead of discarding this tensor, GFM-Hybrid **keeps it and uses it to steer an
entity-augmented BM25 searcher**. Graph and lexical evidence are merged in a
**step-local pool**, a **cross-encoder** does precision ranking, and an **IRCoT**
loop with structured JSON output drives the process across multiple hops.

> The graph supplies **structural breadth**, while entity-augmented lexical matching
> recovers the **surface-level detail** an incomplete graph misses — crucial for
> medical text and low-resource languages such as Vietnamese.

## Headline results (vs. strongest baseline)

| Dataset | Recall@2 | Recall@5 | EM / F1 | LLM-Judge |
|---|---|---|---|---|
| HotpotQA (EN, 2-hop) | **86.75** | **95.65** | **66.10 / 79.57** | — |
| MuSiQue (EN, 2–4 hop) | **56.32** | **74.88** | **41.90 / 52.46** | — |
| PubMedQA (EN, medical) | **65.79** | **86.76** | — | **401/1000** |
| Vietnamese Medical | **98.50** | **98.79** | — | **886/1000** |

---

## 1. Overall Architecture

![GFM-Hybrid overall architecture](assets/figures/pipeline_overview.png)

The interleaved loop runs for up to `max_steps` steps and maintains **four global
memories**: a **global chunk pool** (cleared at the start of each step),
**cumulative facts** (long-term context), **all-discovered-entities** (avoid
repeated lookups), and **previous sub-questions**. Each step runs a retrieval phase
(graph + entity-augmented BM25) and an IRCoT reasoning phase.

### Four components

1. **Graph-Foundation Retriever with Entity Scores** — returns *both* ranked
   documents *and* the min–max-normalised tensor $\tilde{P}_q$; chunks are scored
   with RRF ($k=60$). → `gfmrag_hybrid/gfm/retriever_with_entity_scores.py`
2. **Entity-Augmented BM25 Retrieval** — concatenates seed entities + high-scoring
   graph entities ($\tilde{P}_q \ge \theta=0.10$) + the sub-question into **one**
   BM25 query. → `gfmrag_hybrid/bm25/searcher.py` (`BM25Searcher`)
3. **Step-Local Global Chunk Pool** — merges both branches keeping their scores
   separate, **refreshed every step**. → `core_engine.py`
4. **Cross-Encoder Reranking + IRCoT** — `BAAI/bge-reranker-v2-m3` (max-pooling) +
   an LLM emitting a 5-field JSON object (`reasoning`, `extracted_facts`,
   `missing_entities`, `sub_question`, `final_answer`).
   → `core_engine.py` (`agent_reasoning_with_reranker`)

![Detailed retrieval module](assets/figures/retrieval_module.png)

## 2. Offline Knowledge-Graph Construction

![Offline KG construction](assets/figures/kg_construction.png)

Documents are chunked (LLM/SemanticChunker) → NER + triple extraction →
entity–document matrix → synonym merging. In this repo the offline stage is
organised into **Stage 0** (splitter, bilingual vi/en) and **Stage 1** (optional
chunk grouping + KG building) — see `gfmrag_hybrid/workflow/stage0_split_documents.py` and
`gfmrag_hybrid/kg_construction/chunk_grouper.py`.

---

## 3. Package layout (`gfmrag_hybrid`)

```
gfmrag_hybrid/
├── gfm/
│   └── retriever_with_entity_scores.py        # Component 1 (GFM + entity scores)
├── bm25/
│   ├── searcher.py                            # Component 2 (BM25Searcher, entity-augmented)
│   ├── normalize.py                           # Entity normalisation (normalize_entities)
│   └── stopwords.py                           # vi/en stopwords (VIETNAMESE_STOPWORDS)
├── chunkers/document_chunker.py               # SemanticChunker (splitting)
├── kg_construction/chunk_grouper.py           # Chunk grouping (stage1)
├── utils/text_tokenize.py                     # vi/en tokenisation
└── workflow/
    ├── stage0_split_documents.py              # Splitter
    ├── stage1_index_dataset.py                # Build KG-index
    ├── stage2_kg_pretrain.py / stage2_qa_finetune.py
    ├── stage3_qa_ircot_inference_*.py         # IRCoT inference
    ├── core_engine.py                         # Components 3–4 (pool + rerank + IRCoT)
    ├── app.py                                 # Chatbot (Streamlit)
    └── config/                                # Hydra configs
data/<data_name>/{raw, processed}              # Datasets
gfm_model/                                     # GFM checkpoint
model_cache/                                   # Embedding cache
```

---

## 4. Requirements

| Component | Requirement |
|---|---|
| Python | **3.12** (>=3.12, <3.13) |
| GPU | NVIDIA + **CUDA 12.x** (required for GFM/GNN) |
| LLM API | OpenAI key (OpenAI-compatible endpoint, e.g. Yescale) |

Default models (paper): LLM `GPT-4o-mini`, cross-encoder
`BAAI/bge-reranker-v2-m3`, embedding `Multilingual-E5` (repo config uses
`dangvantuan/vietnamese-embedding` for Vietnamese).

## 5. Installation

```bash
conda create -n gfmhybrid python=3.12 && conda activate gfmhybrid
conda install cuda-toolkit -c nvidia/label/cuda-12.4.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .            # install the gfmrag_hybrid package (editable)
```

Create `.env` in the project root (do NOT commit):

```dotenv
OPENAI_API_KEY=sk-...
HF_TOKEN=hf_...
```

## 6. Data

**Data link:** [Google Drive](https://drive.google.com/file/d/1ILAAFH2UpWpyD9WC1A2eFbjddus2eQ0V/view?usp=drive_link)

Place raw data under `data/<data_name>/raw/`:
- `dataset_corpus.json` — `{ "doc_title": "content..." }`
- `train.json` / `test.json` (optional) — `id`, `question`, `answer`, `supporting_facts`

| Dataset | Domain | Language | Reasoning | Source |
|---|---|---|---|---|
| HotpotQA | General | EN | 2-hop | Wikipedia |
| MuSiQue | General | EN | 2–4 hop | Wikipedia |
| PubMedQA | Medical | EN | Multi-hop | PubMed abstracts |
| Vietnamese Medical | Medical | VI | Multi-hop | Treatment guidelines / pharmacopoeia |

## 7. Usage

```bash
# Stage 0 — split documents -> chunks (bilingual vi/en)
python -m gfmrag_hybrid.workflow.stage0_split_documents \
    dataset.data_name=vietnamese_medical language=vi

# Stage 1 — build KG-index (optionally enable chunk grouping)
python -m gfmrag_hybrid.workflow.stage1_index_dataset \
    dataset.data_name=vietnamese_medical language=vi \
    chunk_grouping.enabled=true chunk_grouping.granularity=chunk

# Stage 2 — (optional) pre-train / fine-tune GFM
python -m gfmrag_hybrid.workflow.stage2_kg_pretrain
python -m gfmrag_hybrid.workflow.stage2_qa_finetune

# Stage 3 — IRCoT inference (GFM-Hybrid)
python -m gfmrag_hybrid.workflow.stage3_qa_ircot_inference_chunks_vietnamese_medical \
    dataset.data_name=vietnamese_medical test.max_steps=3 test.top_k=5

# Web chatbot
cd gfmrag_hybrid/workflow && streamlit run app.py
```

## 8. Hyperparameters (from the paper)

| Parameter | Value | Role |
|---|---|---|
| `top_k` / `top_k_chunks` | 5 / 5 | Graph-branch docs / context chunks to the LLM |
| `max_steps` | 3 | Max IRCoT reasoning steps |
| `top_entity_k` | 15 | High-scoring graph entities kept per step |
| `max_bm25_chunks` / `max_gfm_chunks` | 15 / 20 | Chunk cap per branch |
| `doc_ranker.top_k` | 30 | Entities used for document-ranking weights |
| `k` (RRF) | 60 | RRF smoothing constant |
| `θ` (entity threshold) | 0.10 | Min. $\tilde{P}_q$ for an entity to join the BM25 query |

## 9. Evaluation (LLM-as-a-Judge)

```bash
export OPENAI_API_KEY="sk-..."
python LLM_as_a_judge.py --input prediction.jsonl --output evaluated.jsonl --workers 5
```
Each line in `prediction.jsonl`: `id`, `question`, `answer` (reference), `response`.

GFM-Hybrid extends **GFM-RAG** (Luo et al., NeurIPS 2025) and draws inspiration from
IRCoT, HippoRAG, GraphRAG, LightRAG and the BGE reranker/embedding family.
