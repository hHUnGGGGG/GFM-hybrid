# mypy: ignore-errors
import json
import os
import time
from functools import lru_cache
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING
import math
import torch
from torch import distributed as dist

from gfmrag.ultra import variadic

# Tránh circular import khi chỉ dùng để type hint
if TYPE_CHECKING:
    from gfmrag.gfmrag_retriever_with_entity_scores import EntityScore


class DocumentRetriever:
    """
    Return documents based on document ranking
    """

    def __init__(self, docs: dict, id2doc: dict) -> None:
        self.docs = docs
        self.id2doc = id2doc

    def __call__(self, doc_ranking: torch.Tensor, top_k: int = 1) -> list:
        top_k_docs = doc_ranking.topk(top_k).indices
        norm_doc_scors = mini_max_scale(doc_ranking)
        return [
            {
                "title": self.id2doc[doc.item()],
                "content": self.docs[self.id2doc[doc.item()]],
                "score": doc_ranking[doc].item(),
                "norm_score": norm_doc_scors[doc].item(),
            }
            for doc in top_k_docs
        ]


def mini_max_scale(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())


def entities_to_mask(entities, num_nodes):
    mask = torch.zeros(num_nodes)
    mask[entities] = 1
    return mask


def evaluate(pred, target, metrics):
    ranking, num_pred = pred
    answer_ranking, num_hard = target

    metric = {}
    for _metric in metrics:
        if _metric == "mrr":
            answer_score = 1 / ranking.float()
            query_score = variadic.variadic_mean(answer_score, num_hard)
        elif _metric.startswith("recall@"):
            threshold = int(_metric[7:])
            answer_score = (ranking <= threshold).float()
            query_score = (
                    variadic.variadic_sum(answer_score, num_hard) / num_hard.float()
            )
        elif _metric.startswith("hits@"):
            threshold = int(_metric[5:])
            answer_score = (ranking <= threshold).float()
            query_score = variadic.variadic_mean(answer_score, num_hard)
        elif _metric == "mape":
            query_score = (num_pred - num_hard).abs() / (num_hard).float()
        else:
            raise ValueError(f"Unknown metric `{_metric}`")

        score = query_score.mean()
        name = _metric
        metric[name] = score.item()

    return metric


def gather_results(pred, target, rank, world_size, device):
    # for multi-gpu setups: join results together
    # for single-gpu setups: doesn't do anything special
    ranking, num_pred = pred
    answer_ranking, num_target = target

    all_size_r = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size_ar = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size_p = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size_r[rank] = len(ranking)
    all_size_ar[rank] = len(answer_ranking)
    all_size_p[rank] = len(num_pred)
    if world_size > 1:
        dist.all_reduce(all_size_r, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_size_ar, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_size_p, op=dist.ReduceOp.SUM)

    # obtaining all ranks
    cum_size_r = all_size_r.cumsum(0)
    cum_size_ar = all_size_ar.cumsum(0)
    cum_size_p = all_size_p.cumsum(0)

    all_ranking = torch.zeros(all_size_r.sum(), dtype=torch.long, device=device)
    all_num_pred = torch.zeros(all_size_p.sum(), dtype=torch.long, device=device)
    all_answer_ranking = torch.zeros(all_size_ar.sum(), dtype=torch.long, device=device)
    all_num_target = torch.zeros(all_size_p.sum(), dtype=torch.long, device=device)

    all_ranking[cum_size_r[rank] - all_size_r[rank]: cum_size_r[rank]] = ranking
    all_num_pred[cum_size_p[rank] - all_size_p[rank]: cum_size_p[rank]] = num_pred
    all_answer_ranking[cum_size_ar[rank] - all_size_ar[rank]: cum_size_ar[rank]] = (
        answer_ranking
    )
    all_num_target[cum_size_p[rank] - all_size_p[rank]: cum_size_p[rank]] = num_target

    if world_size > 1:
        dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_pred, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_answer_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_target, op=dist.ReduceOp.SUM)

    return (all_ranking.cpu(), all_num_pred.cpu()), (
        all_answer_ranking.cpu(),
        all_num_target.cpu(),
    )


def batch_evaluate(pred, target, limit_nodes=None):
    num_target = target.sum(dim=-1)

    answer2query = torch.repeat_interleave(num_target)

    num_entity = pred.shape[-1]

    if limit_nodes is not None:
        keep_mask = torch.zeros(num_entity, dtype=torch.bool, device=limit_nodes.device)
        keep_mask[limit_nodes] = 1
        pred[:, ~keep_mask] = float("-inf")

    order = pred.argsort(dim=-1, descending=True)

    range = torch.arange(num_entity, device=pred.device)
    ranking = variadic.native_scatter(
        range.expand_as(order), order, dim=-1, reduce="sum"
    )

    target_ranking = ranking[target]
    order_among_answer = variadic.variadic_sort(target_ranking, num_target)[1]
    order_among_answer = (
            order_among_answer + (num_target.cumsum(0) - num_target)[answer2query]
    )

    ranking_among_answer = variadic.native_scatter(
        variadic.variadic_arange(num_target), order_among_answer, reduce="sum"
    )

    ranking = target_ranking - ranking_among_answer + 1
    ends = num_target.cumsum(0)
    starts = ends - num_target
    hard_mask = variadic.multi_slice_mask(starts, ends, ends[-1])
    ranking = ranking[hard_mask]

    return ranking, target_ranking


# =========================================================================
# BỘ NHỚ ĐỆM (CACHE) — Chỉ nạp JSON một lần duy nhất vào RAM
# =========================================================================
@lru_cache(maxsize=1)
def load_precomputed_chunks(filepath: str) -> dict:
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"[INFO] Đã nạp thành công bộ Precomputed Chunks vào RAM từ {filepath}")
        return data
    else:
        print(f"[ERROR] Không tìm thấy file JSON tại {filepath}. Trả về dict rỗng.")
        return {}


@lru_cache(maxsize=1)
def load_chunk2entities(filepath: str) -> dict:
    if filepath and os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"[INFO] Đã nạp thành công bộ Chunk-to-Entities vào RAM từ {filepath}")
        return data
    else:
        print(f"[WARNING] Không tìm thấy file chunk2entities tại {filepath}. Trả về dict rỗng.")
        return {}


def _resolve_precomputed_path(retriever, precomputed_path: Optional[str]) -> str:
    """Tự động suy ra đường dẫn precomputed_chunks.json nếu không truyền vào."""
    if precomputed_path is not None:
        return precomputed_path
    try:
        return os.path.join(
            retriever.qa_data.root, retriever.qa_data.name,
            "processed", "stage1", "precomputed_chunks.json"
        )
    except AttributeError:
        return "precomputed_chunks.json"


def _chunks_from_ranked_docs(
        ranked_docs: list,
        precomputed_db: dict,
        chunk2entities_db: dict,
        target_entities: List[str] = None,
        entity_scores: list = None,
        global_entity_idf: Dict[str, float] = None,
        max_total_chunks: int = 20,
        rrf_k: int = 60,
) -> List[Dict]:
    if global_entity_idf is None:
        global_entity_idf = {}

    # Bước 1: Khởi tạo chunks và tính rrf_doc
    candidate_chunks: List[Dict] = []
    for rank, doc in enumerate(ranked_docs):
        doc_rank = rank + 1
        rrf_doc = 1.0 / (rrf_k + doc_rank)
        for chunk_info in precomputed_db.get(doc["title"], []):
            candidate_chunks.append({
                "doc_title": doc["title"],
                "doc_score": doc["score"],
                "doc_rank": doc_rank,
                "rrf_doc": rrf_doc,
                "chunk_info": chunk_info,
                "chunk_id": chunk_info.get("chunk_id"),
            })

    if not candidate_chunks:
        return []

    # Bước 2: Build query entities weights (GFM_score)
    query_entities: Dict[str, float] = {}
    if entity_scores:
        for e in entity_scores:
            name = e.entity_name.lower() if hasattr(e, "entity_name") else e.get("entity_name", "").lower()
            score = e.norm_score if hasattr(e, "norm_score") else e.get("norm_score", 0.0)
            if name:
                query_entities[name] = score

    if target_entities:
        for ent in target_entities:
            ent_lower = ent.lower()
            if ent_lower and ent_lower not in query_entities:
                query_entities[ent_lower] = 1.0

    # Nếu không có thực thể, trả về thuần rrf_doc
    if not query_entities:
        result = []
        for item in candidate_chunks:
            ci = item["chunk_info"]
            result.append({
                "text": ci["text"],
                "document_title": item["doc_title"],
                "document_norm_score": item["rrf_doc"],
                "chunk_id": item["chunk_id"],
                "rrf_doc": item["rrf_doc"],
                "rrf_entity": 0.0,
            })
        result.sort(key=lambda x: x["document_norm_score"], reverse=True)
        return result[:max_total_chunks]

    # Bước 3: Tính Entity Score dựa trên IDF kế thừa từ Doc Ranker
    chunk_entities_map = {
        item["chunk_id"]: [e.lower() for e in chunk2entities_db.get(item["chunk_id"], [])]
        for item in candidate_chunks
    }

    for item in candidate_chunks:
        c_ents = chunk_entities_map[item["chunk_id"]]
        raw_score = 0.0
        for ent, gfm_weight in query_entities.items():
            if ent in c_ents:
                doc_idf = global_entity_idf.get(ent, 0.0)
                raw_score += gfm_weight * doc_idf
        item["raw_entity_score"] = raw_score

    # Bước 4: Chuyển raw score thành RRF nội bộ
    chunks_with_ents = [c for c in candidate_chunks if c["raw_entity_score"] > 0]
    chunks_without_ents = [c for c in candidate_chunks if c["raw_entity_score"] == 0]

    chunks_with_ents.sort(key=lambda x: x["raw_entity_score"], reverse=True)

    for rank, item in enumerate(chunks_with_ents):
        item["rrf_entity"] = 1.0 / (rrf_k + rank + 1)

    for item in chunks_without_ents:
        item["rrf_entity"] = 0.0

    # Bước 5: Phép cộng trực tiếp (như yêu cầu)
    result = []
    for item in chunks_with_ents + chunks_without_ents:
        chunk_gfm_rrf = item["rrf_doc"] + item["rrf_entity"]

        result.append({
            "text": item["chunk_info"]["text"],
            "document_title": item["doc_title"],
            "document_score": item["doc_score"],
            "document_norm_score": chunk_gfm_rrf,
            "document_rank": item["doc_rank"],
            "chunk_id": item["chunk_id"],
            "rrf_doc": item["rrf_doc"],
            "rrf_entity": item["rrf_entity"],
        })

    # Bước 6: Sort và cắt cứng đúng số max
    result.sort(key=lambda x: x["document_norm_score"], reverse=True)
    return result[:max_total_chunks]


# =========================================================================
# HÀM TỔNG HỢP ĐIỂM: GFM × BM25-Entity-Weighted
# =========================================================================
def fuse_gfm_and_bm25_chunks(
        gfm_chunks: List[Dict],
        bm25_chunks: List[Dict],
        alpha: float = 0.6,
        beta: float = 0.4,
) -> List[Dict]:
    """
    Hợp nhất GFM chunks (Cascading RRF) và BM25 chunks (Entity-weighted RRF)
    Vì CẢ HAI đều đang ở thang đo RRF tự nhiên, ta KHÔNG chuẩn hóa Min-Max nữa.
    """
    pool: Dict[str, Dict] = {}

    # --- Nạp GFM chunks ---
    for chunk in gfm_chunks:
        from gfmrag.utils.qa_utils import _make_chunk_key  # Đảm bảo import hoặc gọi hàm trực tiếp
        cid = _make_chunk_key(chunk)
        entry = chunk.copy()
        # document_norm_score giờ đây chứa điểm RRF hỗn hợp
        entry["gfm_score"] = float(chunk.get("document_norm_score", 0.0))
        entry["entity_bm25_score"] = 0.0
        pool[cid] = entry

    # --- Nạp BM25 chunks ---
    for chunk in bm25_chunks:
        from gfmrag.utils.qa_utils import _make_chunk_key
        cid = _make_chunk_key(chunk)
        bm25_score = float(chunk.get("entity_weighted_bm25_score",
                                     chunk.get("keyword_score", 0.0)))
        if cid in pool:
            pool[cid]["entity_bm25_score"] = bm25_score
        else:
            entry = chunk.copy()
            entry["gfm_score"] = 0.0
            entry["entity_bm25_score"] = bm25_score
            pool[cid] = entry

    chunks = list(pool.values())

    # --- LƯU Ý: KHÔNG CHUẨN HÓA BM25 MIN-MAX NỮA ---
    # Vì cả gfm_score và entity_bm25_score đều sinh ra từ công thức 1 / (K + rank)
    # nên chúng đã ở cùng một hệ quy chiếu. Cộng trực tiếp!

    # --- Tính combined_score ---
    for c in chunks:
        c["combined_score"] = alpha * c["gfm_score"] + beta * c["entity_bm25_score"]

    return chunks


def _make_chunk_key(chunk: Dict) -> str:
    """
    Tạo key duy nhất để nhận diện chunk khi dedup giữa GFM và BM25.
    Ưu tiên chunk_id; fallback sang title + 80 ký tự đầu nội dung.
    """
    cid = chunk.get("chunk_id", "")
    if cid:
        return str(cid)
    title = chunk.get("document_title", chunk.get("title", ""))
    text_prefix = chunk.get("text", chunk.get("content", ""))[:80]
    return f"{title}::{text_prefix}"


# =========================================================================
# API CHÍNH: Retrieve + BM25 Entity-Aware (dùng GFMRetrieverWithEntityScores)
# =========================================================================
def retrieve_chunks_with_entity_scores(
        retriever,
        entities: List[str],
        top_k: int = 5,
        precomputed_path: Optional[str] = None,
        top_entity_k: int = 30,
        max_total_chunks: int = 20,
) -> Tuple[List[Dict], List]:
    t_start_total = time.time()

    query = " ".join(entities)
    resolved_path = _resolve_precomputed_path(retriever, precomputed_path)
    chunk2entities_path = resolved_path.replace(
        "precomputed_chunks.json", "chunk2entities.json"
    )

    # =========================================================================
    # STEP 1: CORE GFM RETRIEVAL (NER + EL + Embedding + GNN Inference)
    # =========================================================================
    t_gfm_core_start = time.time()
    if hasattr(retriever, "retrieve_with_entity_scores"):
        result = retriever.retrieve_with_entity_scores(
            query=query, top_k=top_k, pre_extracted_entities=entities, top_entity_k=top_entity_k
        )
        ranked_docs = result.docs
        entity_scores = result.top_entity_scores
    else:
        ranked_docs = retriever.retrieve(
            query=query, top_k=top_k, pre_extracted_entities=entities
        )
        entity_scores = []
    t_gfm_core = time.time() - t_gfm_core_start
    print(f"[PROFILER] Step 1 - Core Model (retrieve_with_entity_scores) took: {t_gfm_core:.4f}s")

    # =========================================================================
    # STEP 2: GLOBAL ENTITY IDF CACHING
    # =========================================================================
    t_idf_start = time.time()
    if hasattr(retriever, "_cached_global_entity_idf"):
        global_entity_idf = retriever._cached_global_entity_idf
        print(f"[PROFILER] Step 2 - IDF Caching (Hit cache) took: {time.time() - t_idf_start:.4f}s")
    else:
        global_entity_idf = {}
        if hasattr(retriever, "doc_ranker"):
            ranker = retriever.doc_ranker

            if hasattr(ranker, "idf_weight"):
                idf_list = ranker.idf_weight.detach().cpu().tolist()
            elif hasattr(ranker, "ent2doc"):
                ent2doc = ranker.ent2doc
                if ent2doc.is_sparse:
                    frequency = torch.sparse.sum(ent2doc, dim=-1).to_dense()
                else:
                    frequency = ent2doc.sum(dim=-1)

                idf_tensor = torch.where(
                    frequency > 0, 1.0 / frequency, torch.zeros_like(frequency, dtype=torch.float32)
                )
                idf_list = idf_tensor.detach().cpu().tolist()
            else:
                idf_list = []

            if idf_list:
                id2entity = getattr(retriever, "id2ent", None)
                if id2entity is None and hasattr(retriever, "qa_data"):
                    id2entity = getattr(retriever.qa_data, "id2ent", getattr(retriever.qa_data, "id2entity", []))

                if isinstance(id2entity, dict):
                    for ent_idx, ent_name in id2entity.items():
                        if isinstance(ent_idx, int) and ent_idx < len(idf_list):
                            global_entity_idf[str(ent_name).lower()] = idf_list[ent_idx]
                elif isinstance(id2entity, list):
                    for ent_idx, ent_name in enumerate(id2entity):
                        if ent_idx < len(idf_list):
                            global_entity_idf[str(ent_name).lower()] = idf_list[ent_idx]

        retriever._cached_global_entity_idf = global_entity_idf
        t_idf = time.time() - t_idf_start
        print(f"[PROFILER] Step 2 - IDF Caching (First time build) took: {t_idf:.4f}s")

    # =========================================================================
    # STEP 3: LOAD JSON DICTIONARIES (I/O)
    # =========================================================================
    t_json_start = time.time()
    precomputed_db = load_precomputed_chunks(resolved_path)
    chunk2entities_db = load_chunk2entities(chunk2entities_path)
    t_json = time.time() - t_json_start
    print(f"[PROFILER] Step 3 - Load JSONs (Precomputed + Chunk2Ent) took: {t_json:.4f}s")

    # =========================================================================
    # STEP 4: CHUNKING & RRF MATH
    # =========================================================================
    t_math_start = time.time()
    gfm_chunks = _chunks_from_ranked_docs(
        ranked_docs=ranked_docs,
        precomputed_db=precomputed_db,
        chunk2entities_db=chunk2entities_db,
        target_entities=entities,
        entity_scores=entity_scores,
        global_entity_idf=global_entity_idf,
        max_total_chunks=max_total_chunks,
    )
    t_math = time.time() - t_math_start
    print(f"[PROFILER] Step 4 - Chunk mapping and RRF Math took: {t_math:.4f}s")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    t_total = time.time() - t_start_total
    print(f"[PROFILER] => TOTAL retrieve_chunks_with_entity_scores took: {t_total:.4f}s")

    return gfm_chunks, entity_scores


# =========================================================================
# BACKWARD COMPAT: giữ lại hàm cũ — bây giờ dùng retrieve_chunks_with_entity_scores
# =========================================================================
def retrieve_chunks_with_pre_extracted_entities(
        retriever,
        entities: List[str],
        top_k: int = 5,
        precomputed_path: Optional[str] = None,
) -> List[Dict]:
    """
    API cũ — giữ lại để không break code không cần entity scores.
    Internally gọi retrieve_chunks_with_entity_scores() và bỏ entity_scores đi.
    """
    chunks, _ = retrieve_chunks_with_entity_scores(
        retriever=retriever,
        entities=entities,
        top_k=top_k,
        precomputed_path=precomputed_path,
        top_entity_k=0,  # không cần entity scores → tiết kiệm chi phí build list
    )
    return chunks