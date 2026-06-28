"""
BM25Searcher — nhánh truy hồi từ vựng (lexical) của GFM-Hybrid.

Bản hợp nhất duy nhất, gom từ các bản chép trùng trước đây trong
``workflow/core_engine.py``, ``workflow/stage3_qa_ircot_inference_chunks*.py`` và
``workflow/update.py``. Mỗi điểm gọi sống được bảo toàn hành vi:

- ``search_standard``  — dùng một chuỗi query gộp (core_engine + các stage3 chunks).
- ``search``           — RRF trên nhiều query string (update.py).
- ``search_with_entity_scores`` — RRF có trọng số theo norm_score GNN (update.py).
"""

import json
import string
import logging
from typing import List, Dict, Optional

import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25Searcher:
    def __init__(self, filepath: str, stopwords: set):
        self.stopwords = stopwords
        self.all_chunks: List[Dict] = []
        self.bm25: Optional[BM25Okapi] = None

        logger.info(f"Đang xây dựng BM25 index từ {filepath}...")
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                for chunks in data.values():
                    self.all_chunks.extend(chunks)
            else:
                self.all_chunks = data

            if self.all_chunks:
                corpus_tokens = [
                    self._tokenize(chunk.get("document_title", "") + " " + chunk.get("text", ""))
                    for chunk in self.all_chunks
                ]
                self.bm25 = BM25Okapi(corpus_tokens)
                logger.info(f"BM25 index xây dựng thành công với {len(self.all_chunks)} chunks.")
            else:
                logger.warning("BM25 corpus rỗng.")
        except Exception as e:
            logger.error(f"Không thể xây dựng BM25 index: {e}")

    def _tokenize(self, text: str) -> List[str]:
        text = str(text).lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return [w for w in text.split() if w not in self.stopwords and len(w) > 1]

    def search_standard(self, query: str, top_k: int = 50) -> List[Dict]:
        """Tìm kiếm BM25 tiêu chuẩn với một chuỗi query duy nhất"""
        if not self.bm25 or not self.all_chunks or not query.strip():
            return []

        tokens = self._tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)

        if scores.max() == 0:
            return []

        n_docs = len(scores)
        top_indices = np.argpartition(scores, -min(top_k, n_docs))[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        result = []
        for idx in top_indices:
            score = float(scores[idx])
            if score == 0:
                continue
            chunk = self.all_chunks[int(idx)].copy()
            chunk["bm25_score"] = score
            result.append(chunk)

        return result

    def search(self, queries: List[str], top_k: int = 50) -> List[Dict]:
        """RRF search với nhiều query string."""
        if not self.bm25 or not self.all_chunks or not queries:
            return []
        rrf_scores: Dict[int, float] = {}
        K = 60
        for q in queries:
            query_tokens = self._tokenize(str(q))
            if not query_tokens:
                continue
            doc_scores = self.bm25.get_scores(query_tokens)
            ranked_indices = sorted(
                (i for i, s in enumerate(doc_scores) if s > 0),
                key=lambda i: doc_scores[i],
                reverse=True,
            )
            for rank, idx in enumerate(ranked_indices[:top_k]):
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (K + rank + 1)
        if not rrf_scores:
            return []
        sorted_indices = sorted(
            rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True
        )[:top_k]
        result = []
        for idx in sorted_indices:
            chunk = self.all_chunks[idx].copy()
            chunk["keyword_score"] = rrf_scores[idx]
            chunk["entity_weighted_bm25_score"] = rrf_scores[idx]
            result.append(chunk)
        return result

    def search_with_entity_scores(
            self,
            entity_scores: list,
            top_k: int = 50,
            min_entity_norm_score: float = 0.05,
            base_entities: Optional[List[str]] = None,
            base_entity_weight: float = 1.0,
    ) -> List[Dict]:
        """
        BM25 search dùng TÊN ENTITY làm query, trọng số bởi norm_score GNN.

        Args:
            entity_scores:         List[EntityScore] từ GFM model — mỗi entity
                                   có norm_score dùng làm trọng số RRF.
            top_k:                 Số chunk trả về tối đa.
            min_entity_norm_score: Ngưỡng lọc entity GFM có score quá thấp.
            base_entities:         Init entities (từ NER câu hỏi gốc) — được
                                   search với trọng số đồng đều = base_entity_weight,
                                   không qua ngưỡng min_entity_norm_score.
            base_entity_weight:    Trọng số RRF cho mỗi init entity (mặc định 1.0).
        """
        if not self.bm25 or not self.all_chunks:
            return []
        if not entity_scores and not base_entities:
            return []

        weighted_rrf: Dict[int, float] = {}
        K = 60
        used_entities = 0

        # ── Bước A: Init entities — trọng số đồng đều = base_entity_weight ──
        if base_entities:
            for entity_name in base_entities:
                entity_name = entity_name.strip()
                if not entity_name:
                    continue
                query_tokens = self._tokenize(entity_name)
                if not query_tokens:
                    continue
                doc_scores = self.bm25.get_scores(query_tokens)
                ranked_indices = sorted(
                    (i for i, s in enumerate(doc_scores) if s > 0),
                    key=lambda i: doc_scores[i],
                    reverse=True,
                )
                for rank, idx in enumerate(ranked_indices[:top_k]):
                    weighted_rrf[idx] = (
                            weighted_rrf.get(idx, 0.0) + base_entity_weight / (K + rank + 1)
                    )
            logger.debug(
                f"search_with_entity_scores: {len(base_entities)} base_entities nạp vào RRF "
                f"(weight={base_entity_weight})"
            )

        # ── Bước B: GFM entity scores — trọng số = norm_score GNN ───────────
        for entity in entity_scores:
            if hasattr(entity, "entity_name"):
                entity_name = entity.entity_name
                weight = entity.norm_score
            else:
                entity_name = entity.get("entity_name", "")
                weight = entity.get("norm_score", 0.0)

            if weight < min_entity_norm_score or not entity_name.strip():
                continue

            query_tokens = self._tokenize(entity_name)
            if not query_tokens:
                continue

            doc_scores = self.bm25.get_scores(query_tokens)
            ranked_indices = sorted(
                (i for i, s in enumerate(doc_scores) if s > 0),
                key=lambda i: doc_scores[i],
                reverse=True,
            )

            for rank, idx in enumerate(ranked_indices[:top_k]):
                weighted_rrf[idx] = (
                        weighted_rrf.get(idx, 0.0) + weight / (K + rank + 1)
                )
            used_entities += 1

        logger.debug(
            f"search_with_entity_scores: {used_entities}/{len(entity_scores)} GFM entities dùng được, "
            f"{len(weighted_rrf)} chunks có điểm > 0"
        )

        if not weighted_rrf:
            return []

        sorted_indices = sorted(
            weighted_rrf.keys(), key=lambda i: weighted_rrf[i], reverse=True
        )[:top_k]

        result = []
        for idx in sorted_indices:
            chunk = self.all_chunks[idx].copy()
            score = weighted_rrf[idx]
            chunk["entity_weighted_bm25_score"] = score
            chunk["keyword_score"] = score
            result.append(chunk)
        return result
