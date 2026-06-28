"""
gfmrag_hybrid.gfm — nhánh truy hồi đồ thị (Graph Foundation Model) của GFM-Hybrid.

Gom hai retriever: ``GFMRetriever`` (cơ sở) và ``GFMRetrieverWithEntityScores``
(mở rộng để expose tensor độ liên quan thực thể cho nhánh BM25 hybrid).
"""

from .retriever import GFMRetriever, dedup_retrieved_docs
from .retriever_with_entity_scores import (
    GFMRetrieverWithEntityScores,
    EntityScore,
    RetrieveResult,
)

__all__ = [
    "GFMRetriever",
    "dedup_retrieved_docs",
    "GFMRetrieverWithEntityScores",
    "EntityScore",
    "RetrieveResult",
]
