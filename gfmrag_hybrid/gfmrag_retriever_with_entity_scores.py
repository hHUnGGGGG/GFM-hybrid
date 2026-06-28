"""
gfmrag_retriever_with_entity_scores.py

Subclass của GFMRetriever, override retrieve() để expose entity scores
ra ngoài trong quá trình inference.

Usage (drop-in replacement):
    Thay:  from gfmrag_hybrid import GFMRetriever
    Bằng:  from gfmrag_retriever_with_entity_scores import GFMRetrieverWithEntityScores as GFMRetriever
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from dataclasses import dataclass
from typing import List, Tuple

import torch
from omegaconf import DictConfig

from gfmrag_hybrid import GFMRetriever
from gfmrag_hybrid.ultra import query_utils
from gfmrag_hybrid.utils.qa_utils import entities_to_mask
from gfmrag_hybrid.gfmrag_retriever import dedup_retrieved_docs

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EntityScore:
    """Điểm của một entity sau khi GNN inference."""
    entity_name: str
    entity_id: int
    raw_score: float
    norm_score: float
    rank: int

    def to_dict(self) -> dict:
        return {
            "entity_name": self.entity_name,
            "entity_id": self.entity_id,
            "raw_score": round(self.raw_score, 6),
            "norm_score": round(self.norm_score, 6),
            "rank": self.rank,
        }


@dataclass
class RetrieveResult:
    """Kết quả đầy đủ: documents + entity scores + metadata."""
    docs: list                          # List[Dict] — giống output gốc của retrieve()
    top_entity_scores: List[EntityScore]
    seed_entities: List[str]            # Entities đã qua EL và có trong KG
    query: str

    def to_log_dict(self, max_entities: int = 10) -> dict:
        """Serialize gọn để ghi vào JSONL."""
        return {
            "query": self.query,
            "seed_entities": self.seed_entities,
            "retrieved_doc_titles": [d["title"] for d in self.docs],
            "top_entity_scores": [e.to_dict() for e in self.top_entity_scores[:max_entities]],
        }


# ─────────────────────────────────────────────────────────────────────────────
# SUBCLASS
# ─────────────────────────────────────────────────────────────────────────────

class GFMRetrieverWithEntityScores(GFMRetriever):
    """
    Mở rộng GFMRetriever để expose entity scores trong quá trình inference.

    - retrieve()                    → giống class cha (tương thích 100%)
    - retrieve_with_entity_scores() → trả về RetrieveResult (docs + entity scores)
    - from_config()                 → trả về GFMRetrieverWithEntityScores
    """

    # ── lazy property: invert ent2id → id2ent ────────────────────────────────
    @property
    def id2ent(self) -> dict:
        if not hasattr(self, "_id2ent"):
            self._id2ent = {v: k for k, v in self.qa_data.ent2id.items()}
        return self._id2ent

    # ── helper: tensor → List[EntityScore] ───────────────────────────────────
    def _tensor_to_entity_scores(
        self,
        ent_pred: torch.Tensor,
        top_k: int = 30,
    ) -> List[EntityScore]:
        """
        Chuyển output tensor của GNN → danh sách EntityScore có thể serialize.

        Args:
            ent_pred: shape [1, num_nodes] — output thô của graph_retriever.
            top_k:    Số entity muốn lấy (mặc định 50).
        """
        scores = ent_pred[0].cpu()                   # [num_nodes]
        actual_k = min(top_k, scores.shape[0])

        s_min, s_max = scores.min(), scores.max()
        norm_scores = (scores - s_min) / (s_max - s_min).clamp(min=1e-9)

        top_values, top_indices = scores.topk(actual_k)

        result = []
        for rank, idx in enumerate(top_indices.tolist()):
            result.append(EntityScore(
                entity_name=self.id2ent.get(idx, f"<UNK_{idx}>"),
                entity_id=idx,
                raw_score=scores[idx].item(),
                norm_score=norm_scores[idx].item(),
                rank=rank + 1,
            ))
        return result

    # ── helper: prepare input + trả về seed entities đã qua EL ───────────────
    def _prepare_input_and_return_seeds(
            self,
            query: str,
            pre_extracted_entities: List[str] = None,
    ) -> Tuple[dict, List[str]]:

        # 1. Bypass hoặc chạy NER
        if pre_extracted_entities and len(pre_extracted_entities) > 0:
            mentioned_entities = pre_extracted_entities
        else:
            mentioned_entities = self.ner_model(query)
            if not mentioned_entities:
                mentioned_entities = [query]

        # 2. Xử lý song song Entity Linking và Text Embedding
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_el = executor.submit(self.el_model, mentioned_entities, topk=1)
            future_emb = executor.submit(
                self.text_emb_model.encode,
                [query],
                is_query=True,
                show_progress_bar=False
            )

            linked_entities = future_el.result()
            question_embedding = future_emb.result()

        # 3. Trích xuất Entity IDs và tạo Mask
        seed_entities = [
            ent[0]["entity"]
            for ent in linked_entities.values()
            if ent[0]["entity"] in self.qa_data.ent2id
        ]
        entity_ids = [self.qa_data.ent2id[e] for e in seed_entities]

        question_entities_masks = (
            entities_to_mask(entity_ids, self.num_nodes)
            .unsqueeze(0)
            .to(self.device)
        )

        graph_input = {
            "question_embeddings": question_embedding,
            "question_entities_masks": question_entities_masks,
        }
        return graph_input, seed_entities

    # ── METHOD CHÍNH: trả về RetrieveResult đầy đủ ───────────────────────────
    @torch.no_grad()
    # @torch.no_grad()
    # def retrieve_with_entity_scores(
    #         self,
    #         query: str,
    #         top_k: int,
    #         pre_extracted_entities: List[str] = None,
    #         top_entity_k: int = 30,
    # ) -> RetrieveResult:
    #     """
    #     Phiên bản mở rộng của retrieve() — trả về RetrieveResult (có chèn log Profiler).
    #     """
    #     print("\n[PROFILER] ================= START retrieve_with_entity_scores =================")
    #     t_total_start = time.time()
    #
    #     # ── Bước 1: Chuẩn bị input (NER + EL + Embedding) ──
    #     t_prep_start = time.time()
    #     graph_input, seed_entities = self._prepare_input_and_return_seeds(
    #         query, pre_extracted_entities=pre_extracted_entities
    #     )
    #     t_prep = time.time() - t_prep_start
    #     print(f"[PROFILER] Step A - _prepare_input_and_return_seeds took: {t_prep:.4f}s")
    #
    #     t_cuda_start = time.time()
    #     graph_input = query_utils.cuda(graph_input, device=self.device)
    #     t_cuda = time.time() - t_cuda_start
    #     print(f"[PROFILER] Step B - Moving graph_input to CUDA took: {t_cuda:.4f}s")
    #
    #     # ── Bước 2: GNN Inference ──
    #     t_gnn_start = time.time()
    #     ent_pred = self.graph_retriever(
    #         self.graph, graph_input, entities_weight=self.entities_weight
    #     )
    #     t_gnn = time.time() - t_gnn_start
    #     print(f"[PROFILER] Step C - GNN Inference (graph_retriever) took: {t_gnn:.4f}s")
    #
    #     # ── Bước 3: Document Ranking ──
    #     t_rank_start = time.time()
    #     doc_pred = self.doc_ranker(ent_pred)[0]
    #     t_rank = time.time() - t_rank_start
    #     print(f"[PROFILER] Step D - Doc Ranker (Sparse Matrix Multiplication) took: {t_rank:.4f}s")
    #
    #     t_cpu_start = time.time()
    #     doc_pred_cpu = doc_pred.cpu()
    #     t_cpu = time.time() - t_cpu_start
    #     print(f"[PROFILER] Step E - Moving doc_pred to CPU took: {t_cpu:.4f}s")
    #
    #     t_retriever_start = time.time()
    #     retrieved_docs = self.doc_retriever(doc_pred_cpu, top_k=top_k)
    #     t_retriever = time.time() - t_retriever_start
    #     print(f"[PROFILER] Step F - Document Retriever (Extract & Scale) took: {t_retriever:.4f}s")
    #
    #     t_dedup_start = time.time()
    #     deduped_docs = dedup_retrieved_docs(retrieved_docs)
    #     t_dedup = time.time() - t_dedup_start
    #     print(f"[PROFILER] Step G - Document Deduplication took: {t_dedup:.4f}s")
    #
    #     # ── Bước 4: Trích xuất Entity Scores ──
    #     t_score_start = time.time()
    #     entity_scores = []
    #     if top_entity_k > 0:
    #         entity_scores = self._tensor_to_entity_scores(ent_pred, top_k=top_entity_k)
    #     t_score = time.time() - t_score_start
    #     print(f"[PROFILER] Step H - Extracting Entity Scores (_tensor_to_entity_scores) took: {t_score:.4f}s")
    #
    #     t_total = time.time() - t_total_start
    #     print(f"[PROFILER] => TOTAL retrieve_with_entity_scores executed in: {t_total:.4f}s")
    #     print("[PROFILER] ================= END retrieve_with_entity_scores =================\n")
    #
    #     return RetrieveResult(
    #         docs=deduped_docs[:top_k],
    #         top_entity_scores=entity_scores,
    #         seed_entities=seed_entities,
    #         query=query,
    #     )
    @torch.no_grad()
    def retrieve_with_entity_scores(
            self,
            query: str,
            top_k: int,
            pre_extracted_entities: List[str] = None,
            top_entity_k: int = 30,
    ) -> RetrieveResult:
        """
        Phiên bản mở rộng của retrieve() — trả về RetrieveResult (có chèn log Profiler).
        """
        print("\n[PROFILER] ================= START retrieve_with_entity_scores =================")
        torch.cuda.synchronize()
        t_total_start = time.time()

        # ── Bước 1: Chuẩn bị input (NER + EL + Embedding) ──
        t_prep_start = time.time()
        graph_input, seed_entities = self._prepare_input_and_return_seeds(
            query, pre_extracted_entities=pre_extracted_entities
        )
        t_prep = time.time() - t_prep_start
        print(f"[PROFILER] Step A - _prepare_input_and_return_seeds took: {t_prep:.4f}s")

        t_cuda_start = time.time()
        graph_input = query_utils.cuda(graph_input, device=self.device)
        t_cuda = time.time() - t_cuda_start
        print(f"[PROFILER] Step B - Moving graph_input to CUDA took: {t_cuda:.4f}s")

        # ── Bước 2: GNN Inference (Có Autocast và Đồng bộ GPU) ──
        torch.cuda.synchronize()  # Ép CPU đợi GPU hoàn thành mọi việc trước đó
        t_gnn_start = time.time()

        ent_pred = self.graph_retriever(
            self.graph, graph_input, entities_weight=self.entities_weight
        )

        torch.cuda.synchronize()  # Ép CPU đợi GNN chạy xong để bấm giờ chuẩn
        t_gnn = time.time() - t_gnn_start
        print(f"[PROFILER] Step C - GNN Inference (True GPU Time) took: {t_gnn:.4f}s")

        # ── Bước 3: Document Ranking ──
        torch.cuda.synchronize()
        t_rank_start = time.time()

        doc_pred = self.doc_ranker(ent_pred)[0]

        torch.cuda.synchronize()
        t_rank = time.time() - t_rank_start
        print(f"[PROFILER] Step D - Doc Ranker (True GPU Time) took: {t_rank:.4f}s")

        t_cpu_start = time.time()
        doc_pred_cpu = doc_pred.cpu()
        t_cpu = time.time() - t_cpu_start
        print(f"[PROFILER] Step E - Moving doc_pred to CPU took: {t_cpu:.4f}s")

        t_retriever_start = time.time()
        retrieved_docs = self.doc_retriever(doc_pred_cpu, top_k=top_k)
        t_retriever = time.time() - t_retriever_start
        print(f"[PROFILER] Step F - Document Retriever (Extract & Scale) took: {t_retriever:.4f}s")

        t_dedup_start = time.time()
        deduped_docs = dedup_retrieved_docs(retrieved_docs)
        t_dedup = time.time() - t_dedup_start
        print(f"[PROFILER] Step G - Document Deduplication took: {t_dedup:.4f}s")

        # ── Bước 4: Trích xuất Entity Scores ──
        t_score_start = time.time()
        entity_scores = []
        if top_entity_k > 0:
            entity_scores = self._tensor_to_entity_scores(ent_pred, top_k=top_entity_k)
        t_score = time.time() - t_score_start
        print(f"[PROFILER] Step H - Extracting Entity Scores (_tensor_to_entity_scores) took: {t_score:.4f}s")

        torch.cuda.synchronize()
        t_total = time.time() - t_total_start
        print(f"[PROFILER] => TOTAL retrieve_with_entity_scores executed in: {t_total:.4f}s")
        print("[PROFILER] ================= END retrieve_with_entity_scores =================\n")

        return RetrieveResult(
            docs=deduped_docs[:top_k],
            top_entity_scores=entity_scores,
            seed_entities=seed_entities,
            query=query,
        )
    # ── Override retrieve() để tương thích 100% code cũ ──────────────────────
    @torch.no_grad()
    def retrieve(
        self,
        query: str,
        top_k: int,
        pre_extracted_entities: List[str] = None,
    ) -> list:
        """
        Interface giống class cha. Gọi nội bộ retrieve_with_entity_scores()
        với top_entity_k=0 (không build entity list → không tốn thêm chi phí).
        """
        result = self.retrieve_with_entity_scores(
            query=query,
            top_k=top_k,
            pre_extracted_entities=pre_extracted_entities,
            top_entity_k=0,
        )
        return result.docs

    # ── from_config: trả về GFMRetrieverWithEntityScores ─────────────────────
    @staticmethod
    def from_config(cfg: DictConfig) -> "GFMRetrieverWithEntityScores":
        """
        Drop-in replacement cho GFMRetriever.from_config(cfg).
        Dùng __class__ swap để tái dùng toàn bộ logic init của class cha.
        """
        base: GFMRetriever = GFMRetriever.from_config(cfg)
        base.__class__ = GFMRetrieverWithEntityScores
        return base  # type: ignore[return-value]