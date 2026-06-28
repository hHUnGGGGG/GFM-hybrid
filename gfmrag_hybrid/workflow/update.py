import os
import re

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET_SDK"] = "1"

import io
import json
import logging
import sys
import string
from typing import List, Dict, Optional

from rank_bm25 import BM25Okapi
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import torch
from sentence_transformers import CrossEncoder

# ── Dùng subclass expose entity scores thay vì GFMRetriever gốc ──────────────
from gfmrag_hybrid.gfmrag_retriever_with_entity_scores import GFMRetrieverWithEntityScores

from gfmrag_hybrid.evaluation import RetrievalEvaluator
from gfmrag_hybrid.llms import BaseLanguageModel
from gfmrag_hybrid.prompt_builder import QAPromptBuilder
from gfmrag_hybrid.ultra import query_utils
from gfmrag_hybrid.utils.qa_utils import (
    retrieve_chunks_with_entity_scores,
    fuse_gfm_and_bm25_chunks,
)

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

logger = logging.getLogger(__name__)

# =========================================================================
# BỘ LỌC STOPWORDS TIẾNG VIỆT
# =========================================================================
VIETNAMESE_STOPWORDS = {
    "và", "là", "của", "có", "trong", "về", "cách", "nó", "thể", "hỗ", "trợ",
    "trường", "hợp", "do", "quá", "liều", "các", "những", "cho", "để", "với",
    "không", "khi", "được", "một", "này", "đó", "thuốc", "bệnh", "sự", "bị",
    "ra", "vào", "tôi", "cần", "tìm", "thêm", "thông", "tin", "chi", "tiết",
    "làm", "sao", "như", "thế", "nào", "thực", "thể", "thiếu", "tương", "tác"
}


# =========================================================================
# CHUẨN HÓA TÊN THỰC THỂ
# =========================================================================
def normalize_entity(entity: str) -> str:
    e = entity.strip()
    e = re.sub(r'^\[(.+)\]$', r'\1', e).strip()
    return e if e else entity.strip()


def normalize_entities(entities: list) -> list:
    seen = set()
    result = []
    for raw in entities:
        cleaned = normalize_entity(str(raw))
        key = cleaned.lower()
        if key not in seen and cleaned:
            seen.add(key)
            result.append(cleaned)
    return result


# =========================================================================
# CLASS: BM25 SEARCHER — hỗ trợ entity-weighted search
# =========================================================================
class BM25Searcher:
    """
    BM25 index trên precomputed chunks.

    Hai chế độ search:
      - search()                    : multi-query RRF (như cũ)
      - search_with_entity_scores() : mỗi entity dùng norm_score làm trọng số RRF,
                                      để chunks liên quan đến entities có điểm GNN cao
                                      được ưu tiên hơn.
    """

    def __init__(self, filepath: str, stopwords: set):
        self.stopwords = stopwords
        self.all_chunks: List[Dict] = []
        self.bm25: Optional[BM25Okapi] = None
        logger.info(f"Đang xây dựng BM25 index từ {filepath}...")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                for chunks in data.values():
                    self.all_chunks.extend(chunks)
            else:
                self.all_chunks = data
            if self.all_chunks:
                corpus_tokens = [
                    self._tokenize(
                        chunk.get("document_title", "") + " " + chunk.get("text", "")
                    )
                    for chunk in self.all_chunks
                ]
                self.bm25 = BM25Okapi(corpus_tokens)
                logger.info(
                    f"BM25 index xây dựng thành công với {len(self.all_chunks)} chunks."
                )
            else:
                logger.warning("BM25 corpus rỗng.")
        except Exception as e:
            logger.error(f"Không thể xây dựng BM25 index: {e}")

    def _tokenize(self, text: str) -> List[str]:
        text = str(text).lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return [w for w in text.split() if w not in self.stopwords and len(w) > 1]

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


# =========================================================================
# AGENT REASONING CORE
# =========================================================================
def agent_reasoning_with_reranker(
        cfg: DictConfig,
        gfmrag_retriever: GFMRetrieverWithEntityScores,
        reranker: CrossEncoder,
        llm: BaseLanguageModel,
        qa_prompt_builder: QAPromptBuilder,
        query: str,
        bm25_searcher: Optional[BM25Searcher] = None,
) -> dict:
    step = 1
    current_query = query
    all_thoughts: List[str] = []
    logs = []

    precomputed_path = cfg.get("precomputed_chunks_path", None)

    alpha = float(cfg.test.get("fusion_alpha", 0.6))
    beta = float(cfg.test.get("fusion_beta", 0.4))
    top_entity_k = int(cfg.test.get("top_entity_k", 30))
    # --- THÊM: Đọc tham số max_fused_chunks từ config ---
    max_fused_chunks = int(cfg.test.get("max_fused_chunks", 60))

    # ── Bước 0: NER vòng đầu ──────────────────────────────────────────────────
    raw_entities = gfmrag_retriever.ner_model(current_query)
    entities = normalize_entities(raw_entities)
    logger.info(f"Initial entities for retrieval: {entities}")

    all_discovered_entities = set(e.lower() for e in entities)
    global_chunk_pool: Dict[str, Dict] = {}
    all_sub_questions: List[str] = []

    def fetch_and_fuse_into_pool(
            query_entities: List[str],
            extra_bm25_queries: Optional[List[str]] = None,
            label: str = "",
    ) -> int:
        if not query_entities:
            return 0

        # Đọc cấu hình max chunks từ file yaml (nếu không có mặc định là 5)
        max_chunks_per_doc = int(cfg.test.get("max_chunks_per_doc", 10))

        # --- Bước 1: GFM → chunks + entity scores ---
        gfm_chunks, entity_scores = retrieve_chunks_with_entity_scores(
            retriever=gfmrag_retriever,
            entities=query_entities,  # Đây chính là target_entities sẽ được truyền xuống
            top_k=cfg.test.top_k,
            precomputed_path=precomputed_path,
            top_entity_k=top_entity_k,
            max_chunks_per_doc=max_chunks_per_doc  # SỬA DÒNG NÀY: Truyền tham số giới hạn xuống
        )

        logger.info(
            f"{label} GFM {len(gfm_chunks)} chunks, "
            f"{len(entity_scores)} entity scores (top: "
            + (f"{entity_scores[0].entity_name}={entity_scores[0].norm_score:.3f}" if entity_scores else "none")
            + ")"
        )

        # --- Bước 2: BM25 entity-weighted search (Init entities + GFM entity scores) ---
        bm25_entity_chunks: List[Dict] = []
        if bm25_searcher and (entity_scores or query_entities):
            bm25_entity_chunks = bm25_searcher.search_with_entity_scores(
                entity_scores=entity_scores,
                top_k=cfg.test.top_k * 2,
                base_entities=query_entities,  # init entities với weight đồng đều
                base_entity_weight=1.0,
            )
            logger.info(
                f"{label} BM25 entity-weighted {len(bm25_entity_chunks)} chunks "
                f"(base={len(query_entities)} init entities + "
                f"{len(entity_scores)} GFM entity scores)"
            )

        # --- Bước 3: BM25 keyword search bổ sung ---
        # Mỗi entity là một query riêng trong RRF (không join thành 1 chuỗi)
        # để tránh dilute điểm BM25 khi có nhiều entity cùng lúc.
        bm25_keyword_chunks: List[Dict] = []
        if bm25_searcher:
            keyword_queries: List[str] = []

            # (a) Extra queries (sub_q / missing entities string) — giữ nguyên
            if extra_bm25_queries:
                keyword_queries.extend(extra_bm25_queries)

            # (b) Từng init entity là một query riêng
            keyword_queries.extend(query_entities)

            # (c) Top GFM entity names có norm_score >= 0.1 — mỗi cái là query riêng
            if entity_scores:
                top_gfm_names = [
                    (e.entity_name if hasattr(e, "entity_name") else e.get("entity_name", ""))
                    for e in entity_scores
                    if (e.norm_score if hasattr(e, "norm_score") else e.get("norm_score", 0.0)) >= 0.1
                ]
                keyword_queries.extend(top_gfm_names)

            # Loại bỏ chuỗi rỗng và trùng lặp, giữ thứ tự
            keyword_queries = list(dict.fromkeys(q.strip() for q in keyword_queries if q.strip()))

            if keyword_queries:
                bm25_keyword_chunks = bm25_searcher.search(
                    keyword_queries, top_k=cfg.test.top_k
                )
                logger.info(
                    f"{label} BM25 keyword extent {len(bm25_keyword_chunks)} chunks "
                    f"({len(keyword_queries)} queries)"
                )

        # --- Bước 4: Fusion GFM + BM25-entity + BM25-keyword ---
        fused_chunks = fuse_gfm_and_bm25_chunks(
            gfm_chunks=gfm_chunks,
            bm25_chunks=bm25_entity_chunks + bm25_keyword_chunks,
            alpha=alpha,
            beta=beta,
        )

        # --- THÊM: Sắp xếp giảm dần theo điểm và lấy tối đa số chunks từ config ---
        fused_chunks = sorted(fused_chunks, key=lambda x: x.get("combined_score", 0.0), reverse=True)[:max_fused_chunks]

        # --- Bước 5: Đưa vào pool ---
        added = 0
        for c in fused_chunks:
            from gfmrag_hybrid.utils.qa_utils import _make_chunk_key
            cid = _make_chunk_key(c)
            if cid not in global_chunk_pool:
                global_chunk_pool[cid] = c
                added += 1
            else:
                if c.get("combined_score", 0) > global_chunk_pool[cid].get("combined_score", 0):
                    global_chunk_pool[cid] = c

        logger.info(
            f"{label} Fused {len(fused_chunks)} chunks → add {added} into pool "
            f"(pool size: {len(global_chunk_pool)})"
        )
        return added

    is_multi_hop = len(entities) >= 2
    extra_init = [" ".join(entities), current_query] if is_multi_hop else None
    fetch_and_fuse_into_pool(
        query_entities=entities,
        extra_bm25_queries=extra_init,
        label="[Step-0]",
    )

    if not global_chunk_pool and bm25_searcher:
        logger.warning("[Step-0] Pool empty GFM, fallback BM25 keyword...")
        fallback = bm25_searcher.search([current_query] + entities, top_k=cfg.test.top_k * 2)
        for c in fallback:
            from gfmrag_hybrid.utils.qa_utils import _make_chunk_key
            cid = _make_chunk_key(c)
            if cid not in global_chunk_pool:
                global_chunk_pool[cid] = c

    # =========================================================================
    # RERANK POOL
    # =========================================================================
    def rerank_pool(target_queries: List[str]) -> List[Dict]:
        pool_docs = list(global_chunk_pool.values())

        # ---> DÒNG LOG ĐƯỢC THÊM VÀO ĐÂY <---
        logger.info(
            f"[Rerank] Reranker. chunks in global_chunk_pool: {len(pool_docs)}")

        if not pool_docs:
            return []

        valid_queries = [q for q in target_queries if q and q.strip()] or [current_query]

        for chunk in pool_docs:
            chunk["max_score"] = -999.0

        for q in valid_queries:
            pairs = [
                [q, f"Tiêu đề: {c.get('document_title', c.get('title', 'Unknown'))} "
                    f"| Nội dung: {c.get('text', c.get('content', ''))}"]
                for c in pool_docs
            ]
            scores = reranker.predict(pairs, batch_size=64)
            for i, score in enumerate(scores):
                if float(score) > pool_docs[i]["max_score"]:
                    pool_docs[i]["max_score"] = float(score)

        ranked = []
        for chunk in pool_docs:
            ranked.append({
                "title": chunk.get("document_title", chunk.get("title", "Unknown")),
                "content": chunk.get("text", chunk.get("content", "")),
                "score": chunk["max_score"],
                "chunk_id": chunk.get("chunk_id", "Unknown"),
                "gfm_score": chunk.get("gfm_score", 0.0),
                "entity_bm25_score": chunk.get("entity_bm25_score", 0.0),
                "combined_score": chunk.get("combined_score", 0.0),
            })

        return sorted(ranked, key=lambda x: x["score"], reverse=True)

    retrieved_docs = rerank_pool([current_query])

    found_final_answer = None
    cumulative_facts: Dict = {}
    previous_sub_questions: set = set()

    # =========================================================================
    # VÒNG LẶP IRCoT
    # =========================================================================
    while step <= cfg.test.max_steps:
        logger.info(f"\n--- Bước {step} ---")

        docs_to_llm = retrieved_docs[:cfg.test.top_k_chunks]

        memory_str = json.dumps(cumulative_facts, ensure_ascii=False) if cumulative_facts else "{}"

        message = qa_prompt_builder.build_input_prompt(
            current_query,
            docs_to_llm,
            [f"Kiểm kê sự kiện đã xác nhận (tất cả bước trước): {memory_str}"]
        )

        logger.info("Gọi LLM để suy luận JSON...")
        raw_response = llm.generate_sentence(message)

        try:
            json_match = re.search(r'(\{.*\})', raw_response, re.DOTALL)
            if json_match:
                response_json = json.loads(json_match.group(1))
            else:
                response_json = json.loads(raw_response)
        except Exception:
            logger.error("Không thể parse JSON từ LLM.")
            response_json = {
                "extracted_facts": {},
                "missing_entities_to_search": entities,
                "final_answer": None,
            }

        all_thoughts.append(raw_response)

        if "inventory" in response_json:
            response_json["extracted_facts"] = response_json.pop("inventory")

        current_facts = response_json.get("extracted_facts", {})
        if isinstance(current_facts, dict):
            for key, val in current_facts.items():
                if val and str(val).lower() not in ["false", "none", "null"]:
                    cumulative_facts[key] = val

        sub_q = (response_json.get("sub_question") or "").strip() or None

        if sub_q:
            all_sub_questions.append(sub_q)
            previous_sub_questions.add(sub_q)
            logger.info(f"Sub-Question: {sub_q}")

        ner_entities: List[str] = []
        if sub_q:
            try:
                raw_from_ner = gfmrag_retriever.ner_model(sub_q)
                ner_entities = normalize_entities(
                    [raw_from_ner] if isinstance(raw_from_ner, str) else (raw_from_ner or [])
                )
            except Exception as e:
                logger.warning(f"NER thất bại trên sub_q: {e}")

        json_missing = response_json.get("missing_entities_to_search", [])
        json_entities = normalize_entities([str(e) for e in json_missing if e]) if json_missing else []

        merged_for_gfm = list(dict.fromkeys(ner_entities + json_entities))
        last_missing_entities = [e for e in merged_for_gfm if e.lower() not in all_discovered_entities]

        if merged_for_gfm:
            logger.info(f"GFM entities (NER+JSON): {merged_for_gfm}")
        if last_missing_entities:
            logger.info(f"New entities BM25/tracking: {last_missing_entities}")

        found_final_answer = response_json.get("final_answer")

        logs.append({
            "step": step,
            "query": current_query,
            "retrieved_docs": docs_to_llm,
            "response": response_json,
            "extracted_entities": merged_for_gfm,
            "cumulative_facts": cumulative_facts.copy(),
        })

        if found_final_answer:
            logger.info(f"Đã tìm thấy final_answer ở bước {step}")
            break

        step += 1

        if merged_for_gfm:
            all_discovered_entities.update(e.lower() for e in merged_for_gfm)

            extra_queries: List[str] = []
            if last_missing_entities:
                extra_queries.append(" ".join(last_missing_entities))
            if sub_q:
                extra_queries.append(sub_q)

            added = fetch_and_fuse_into_pool(
                query_entities=merged_for_gfm,
                extra_bm25_queries=extra_queries or None,
                label=f"[Step-{step}]",
            )

            if sub_q:
                retrieved_docs = rerank_pool([sub_q])
            else:
                retrieved_docs = rerank_pool([current_query])

        elif sub_q and bm25_searcher:
            logger.info(f"Không có entities, BM25 keyword fallback với sub_q: '{sub_q}'")
            sq_chunks = bm25_searcher.search([sub_q], top_k=cfg.test.top_k)
            added = 0
            for c in sq_chunks:
                from gfmrag_hybrid.utils.qa_utils import _make_chunk_key
                cid = _make_chunk_key(c)
                if cid not in global_chunk_pool:
                    global_chunk_pool[cid] = c
                    added += 1
            logger.info(f"BM25 sub_q fallback thêm {added} chunks")
            retrieved_docs = rerank_pool([sub_q])
        else:
            retrieved_docs = rerank_pool([current_query])

    final_retrieved_docs = retrieved_docs

    final_output = found_final_answer or "Tài liệu hiện tại không đủ thông tin để kết luận."

    if isinstance(final_output, str):
        if final_output.startswith("Vậy câu trả lời là:"):
            final_output = final_output.replace("Vậy câu trả lời là:", "").strip()
    else:
        final_output = json.dumps(final_output, ensure_ascii=False)

    return {
        "response": final_output,
        "retrieved_docs": final_retrieved_docs[:cfg.test.top_k_chunks],
        "logs": logs,
    }


# =========================================================================
# HYDRA MAIN
# =========================================================================
@hydra.main(
    config_path="config",
    config_name="stage3_qa_ircot_inference_chunks",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    output_dir = HydraConfig.get().runtime.output_dir

    try:
        logger.info(f"Config:\n {OmegaConf.to_yaml(cfg)}")
    except UnicodeEncodeError:
        logger.info("Config loaded (unicode log skipped on Windows)")

    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Output directory: {output_dir}")

    gfmrag_retriever = GFMRetrieverWithEntityScores.from_config(cfg)
    logger.info(f"[INFO] Retriever type: {type(gfmrag_retriever).__name__}")
    logger.info(f"[INFO] Document Ranker (Stage 1): {type(gfmrag_retriever.doc_ranker).__name__}")

    llm = instantiate(cfg.llm)

    precomputed_path = cfg.get("precomputed_chunks_path", None)
    bm25_searcher = None
    if precomputed_path and os.path.exists(precomputed_path):
        bm25_searcher = BM25Searcher(precomputed_path, VIETNAMESE_STOPWORDS)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Loading Cross-Encoder Reranker on {device.upper()} (FP16 mode)...")

    reranker = CrossEncoder(
        'BAAI/bge-reranker-v2-m3',
        device=device,
        model_kwargs={"torch_dtype": torch.float16},
    )

    try:
        model_name = reranker.model.config._name_or_path
    except AttributeError:
        model_name = 'CrossEncoder'

    logger.info(f"[INFO] Chunk Reranker (Stage 2): {model_name}")

    agent_prompt_builder = QAPromptBuilder(cfg.agent_prompt)

    test_data = gfmrag_retriever.qa_data.raw_test_data
    max_samples = (
        cfg.test.max_test_samples if cfg.test.max_test_samples > 0 else len(test_data)
    )
    logger.info(f"Total test samples: {len(test_data)}, running: {max_samples}")

    processed_data = {}
    if cfg.test.resume:
        logger.info(f"Resuming from {cfg.test.resume}")
        try:
            with open(cfg.test.resume, encoding="utf-8") as f:
                for line in f:
                    result = json.loads(line)
                    processed_data[result["id"]] = result
            logger.info(f"Loaded {len(processed_data)} processed samples")
        except Exception as e:
            logger.error(f"Could not resume: {e}")

    prediction_path = os.path.join(output_dir, "prediction.jsonl")
    with open(prediction_path, "w", encoding="utf-8") as f:
        for i in tqdm(range(max_samples), desc="Inference"):
            if i >= len(test_data):
                break

            sample = test_data[i]
            query = sample["question"]

            if sample["id"] in processed_data:
                result = processed_data[sample["id"]]
            else:
                try:
                    result_dict = agent_reasoning_with_reranker(
                        cfg,
                        gfmrag_retriever,
                        reranker,
                        llm,
                        agent_prompt_builder,
                        query,
                        bm25_searcher,
                    )
                    result = {
                        "id": sample["id"],
                        "question": sample["question"],
                        "answer": sample["answer"],
                        "answer_aliases": sample.get("answer_aliases", []),
                        "supporting_facts": sample["supporting_facts"],
                        "response": result_dict["response"],
                        "retrieved_docs": result_dict["retrieved_docs"],
                        "logs": result_dict["logs"],
                    }
                except Exception as e:
                    logger.error(f"Error at sample {i} (id={sample.get('id')}): {e}")
                    result = {
                        "id": sample.get("id", f"error_{i}"),
                        "question": query,
                        "answer": sample.get("answer", ""),
                        "answer_aliases": sample.get("answer_aliases", []),
                        "supporting_facts": sample.get("supporting_facts", []),
                        "response": "ERROR",
                        "retrieved_docs": [],
                        "logs": [],
                        "error": str(e),
                    }

            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

    logger.info(f"Predictions saved to {prediction_path}")

    try:
        evaluator = instantiate(cfg.qa_evaluator, prediction_file=prediction_path)
        metrics = evaluator.evaluate()
        query_utils.print_metrics(metrics, logger)
    except Exception as e:
        logger.error(f"QA evaluation error: {e}")

    try:
        retrieval_evaluator = RetrievalEvaluator(prediction_file=prediction_path)
        retrieval_metrics = retrieval_evaluator.evaluate()
        query_utils.print_metrics(retrieval_metrics, logger)
    except Exception as e:
        logger.error(f"Retrieval evaluation error: {e}")


if __name__ == "__main__":
    main()