"""
core_engine.py
==============
Tách toàn bộ logic xử lý từ stage3_qa_ircot_inference_chunks.py.
File này KHÔNG chứa @hydra.main — chỉ export các hàm/class để app.py dùng.
Hệ thống 2 nhánh độc lập GFM và BM25 (Early Fusion) đẩy thẳng vào pool chờ Cross-Encoder làm trọng tài.
Yêu cầu:
- Xóa sạch global_chunk_pool sau mỗi step để tập trung hoàn toàn vào sub_question.
- Nếu có sub_question và chưa max step, bắt buộc đi tiếp (bỏ qua final_answer ảo).
"""

import re
import json
import logging
import time
from typing import List, Dict, Optional

from omegaconf import DictConfig
from sentence_transformers import CrossEncoder

from gfmrag_hybrid.bm25 import BM25Searcher, normalize_entities
from gfmrag_hybrid.gfm.retriever_with_entity_scores import GFMRetrieverWithEntityScores
from gfmrag_hybrid.llms import BaseLanguageModel
from gfmrag_hybrid.prompt_builder import QAPromptBuilder
from gfmrag_hybrid.utils.qa_utils import (
    retrieve_chunks_with_entity_scores,
)

logger = logging.getLogger(__name__)


# =========================================================================
# HELPER: FORMAT DICT/LIST → PROSE (Cho UI)
# =========================================================================
def _dict_to_prose(d, indent=0) -> str:
    if not d: return "Không có thông tin."
    lines = []
    spaces = "  " * indent
    if isinstance(d, dict):
        for key, value in d.items():
            clean_key = str(key).replace("_", " ").capitalize()
            if isinstance(value, (dict, list)):
                lines.append(f"{spaces}- **{clean_key}:**")
                lines.append(_dict_to_prose(value, indent + 1))
            else:
                lines.append(f"{spaces}- **{clean_key}:** {value}")
    elif isinstance(d, list):
        for item in d:
            if isinstance(item, (dict, list)):
                lines.append(f"{spaces}-")
                lines.append(_dict_to_prose(item, indent + 1))
            else:
                lines.append(f"{spaces}- {item}")
    else:
        return f"{spaces}{d}"
    return "\n".join(lines)


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
        use_bm25: bool = True,
) -> dict:
    time_total_start = time.time()
    total_time_ner, total_time_gfm, total_time_bm25, total_time_rerank, total_time_llm = 0.0, 0.0, 0.0, 0.0, 0.0

    step = 1
    current_query = query
    all_thoughts: List[str] = []
    logs = []

    precomputed_path = cfg.get("precomputed_chunks_path", None)
    top_entity_k = int(cfg.test.get("top_entity_k", 30))

    # --- Bước 0: NER vòng đầu ---
    t_ner = time.time()
    raw_entities = gfmrag_retriever.ner_model(current_query)
    entities = normalize_entities(raw_entities)
    total_time_ner += (time.time() - t_ner)
    logger.info(f"Initial entities for retrieval: {entities}")

    all_discovered_entities = set(e.lower() for e in entities)
    global_chunk_pool: Dict[str, Dict] = {}
    all_sub_questions: List[str] = []
    previous_sub_questions: set = set()

    def fetch_and_fuse_into_pool(
            query_entities: List[str],
            extra_bm25_queries: Optional[List[str]] = None,
            label: str = "",
    ) -> int:
        nonlocal total_time_gfm, total_time_bm25
        if not query_entities:
            return 0

        max_gfm_chunks = int(cfg.test.get("max_gfm_chunks", 20))
        max_bm25_chunks = int(cfg.test.get("max_bm25_chunks", 20))

        # =========================================================
        # NHÁNH 1: TRUY XUẤT GFM
        # =========================================================
        t_gfm_start = time.time()
        gfm_chunks, entity_scores = retrieve_chunks_with_entity_scores(
            retriever=gfmrag_retriever,
            entities=query_entities,
            top_k=cfg.test.top_k,
            precomputed_path=precomputed_path,
            top_entity_k=top_entity_k,
            max_total_chunks=max_gfm_chunks
        )
        total_time_gfm += (time.time() - t_gfm_start)

        logger.info(
            f"{label} Nhánh GFM: Trả về {len(gfm_chunks)} chunks, "
            f"{len(entity_scores)} entity scores (top: "
            + (f"{entity_scores[0].entity_name}={entity_scores[0].norm_score:.3f}" if entity_scores else "none")
            + ")"
        )

        # =========================================================
        # NHÁNH 2: TRUY XUẤT BM25 (GỘP CHUỖI)
        # =========================================================
        t_bm25_start = time.time()
        top_bm25_chunks: List[Dict] = []
        if use_bm25 and bm25_searcher:
            combined_terms = [e for e in query_entities if e.strip()]
            base_set = set(e.lower().strip() for e in combined_terms)

            if entity_scores:
                for e in entity_scores:
                    name = e.entity_name if hasattr(e, "entity_name") else e.get("entity_name", "")
                    name_lower = name.lower().strip()
                    score = e.norm_score if hasattr(e, "norm_score") else e.get("norm_score", 0.0)
                    if name_lower and score >= 0.15 and name_lower not in base_set:
                        combined_terms.append(name)
                        base_set.add(name_lower)

            if extra_bm25_queries:
                for eq in extra_bm25_queries:
                    eq_lower = eq.lower().strip()
                    if eq_lower and eq_lower not in base_set:
                        combined_terms.append(eq.strip())
                        base_set.add(eq_lower)

            combined_query_string = " ".join(combined_terms)
            logger.info(f"{label} BM25 Combined Query: '{combined_query_string}'")

            top_bm25_chunks = bm25_searcher.search_standard(
                query=combined_query_string,
                top_k=max_bm25_chunks
            )
            logger.info(f"{label} Nhánh BM25: Đã lấy chính xác {len(top_bm25_chunks)} chunks (Giới hạn: {max_bm25_chunks}).")

        total_time_bm25 += (time.time() - t_bm25_start)

        # =========================================================
        # BƯỚC 3: ĐẨY HAI NHÁNH VÀO GLOBAL POOL
        # =========================================================
        added = 0
        for c in gfm_chunks:
            from gfmrag_hybrid.utils.qa_utils import _make_chunk_key
            cid = _make_chunk_key(c)
            if cid not in global_chunk_pool:
                global_chunk_pool[cid] = c.copy()
                added += 1
            else:
                for k in ["rrf_doc", "rrf_entity", "document_norm_score"]:
                    if k in c:
                        global_chunk_pool[cid][k] = c[k]

        for c in top_bm25_chunks:
            from gfmrag_hybrid.utils.qa_utils import _make_chunk_key
            cid = _make_chunk_key(c)
            if cid not in global_chunk_pool:
                global_chunk_pool[cid] = c.copy()
                added += 1
            else:
                global_chunk_pool[cid]["bm25_score"] = c.get("bm25_score", 0.0)

        logger.info(
            f"{label} Bơm {added} chunks vào pool chờ Cross-Encoder "
            f"(Current Pool size: {len(global_chunk_pool)})"
        )
        return added

    is_multi_hop = len(entities) >= 2
    extra_init = [" ".join(entities), current_query] if is_multi_hop else None
    fetch_and_fuse_into_pool(query_entities=entities, extra_bm25_queries=extra_init, label="[Step-0]")

    if not global_chunk_pool and (use_bm25 and bm25_searcher):
        logger.warning("[Step-0] Pool empty GFM, fallback BM25 keyword...")
        t_bm25_fb = time.time()
        combined_fallback = " ".join([current_query] + entities)
        fallback = bm25_searcher.search_standard(combined_fallback, top_k=cfg.test.top_k * 2)
        for c in fallback:
            from gfmrag_hybrid.utils.qa_utils import _make_chunk_key
            cid = _make_chunk_key(c)
            if cid not in global_chunk_pool:
                c_copy = c.copy()
                c_copy["bm25_score"] = float(c.get("bm25_score", 0.0))
                global_chunk_pool[cid] = c_copy
        total_time_bm25 += (time.time() - t_bm25_fb)

    def rerank_pool(target_queries: List[str]) -> List[Dict]:
        nonlocal total_time_rerank
        t_rerank = time.time()
        pool_docs = list(global_chunk_pool.values())

        logger.info(f"[Rerank] Reranker. chunks in global_chunk_pool: {len(pool_docs)}")

        if not pool_docs:
            total_time_rerank += (time.time() - t_rerank)
            return []

        valid_queries = [q for q in target_queries if q and q.strip()] or [current_query]
        for chunk in pool_docs: chunk["max_score"] = -999.0

        for q in valid_queries:
            pairs = [[q, f"Tiêu đề: {c.get('document_title', c.get('title', 'Unknown'))} | Nội dung: {c.get('text', c.get('content', ''))}"] for c in pool_docs]
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
                "rrf_doc": chunk.get("rrf_doc", 0.0),
                "rrf_entity": chunk.get("rrf_entity", 0.0),
                "document_norm_score": chunk.get("document_norm_score", 0.0),
                "bm25_score": chunk.get("bm25_score", 0.0),
            })
        res = sorted(ranked, key=lambda x: x["score"], reverse=True)
        total_time_rerank += (time.time() - t_rerank)
        return res

    retrieved_docs = rerank_pool([current_query])
    found_final_answer = None
    cumulative_facts: Dict = {}

    while step <= cfg.test.max_steps:
        logger.info(f"\n--- Bước {step} ---")
        docs_to_llm = retrieved_docs[:cfg.test.top_k_chunks]
        memory_str = json.dumps(cumulative_facts, ensure_ascii=False) if cumulative_facts else "{}"
        message = qa_prompt_builder.build_input_prompt(current_query, docs_to_llm, [f"Kiểm kê sự kiện đã xác nhận (tất cả bước trước): {memory_str}"])

        logger.info("Gọi LLM để suy luận JSON...")
        t_llm = time.time()
        raw_response = llm.generate_sentence(message)
        total_time_llm += (time.time() - t_llm)

        try:
            json_match = re.search(r'(\{.*\})', raw_response, re.DOTALL)
            if json_match:
                response_json = json.loads(json_match.group(1))
            else:
                response_json = json.loads(raw_response)
        except Exception:
            logger.error("Không thể parse JSON từ LLM.")
            response_json = {"extracted_facts": {}, "missing_entities_to_search": entities, "final_answer": None}

        all_thoughts.append(raw_response)
        if "inventory" in response_json: response_json["extracted_facts"] = response_json.pop("inventory")

        current_facts = response_json.get("extracted_facts", {})
        if isinstance(current_facts, dict):
            for key, val in current_facts.items():
                if val and str(val).lower() not in ["false", "none", "null"]: cumulative_facts[key] = val

        sub_q = (response_json.get("sub_question") or "").strip() or None

        if sub_q:
            all_sub_questions.append(sub_q)
            previous_sub_questions.add(sub_q)
            logger.info(f"Sub-Question: {sub_q}")

        ner_entities: List[str] = []
        if sub_q:
            t_ner_sub = time.time()
            try:
                raw_from_ner = gfmrag_retriever.ner_model(sub_q)
                ner_entities = normalize_entities([raw_from_ner] if isinstance(raw_from_ner, str) else (raw_from_ner or []))
            except Exception as e:
                logger.warning(f"NER thất bại trên sub_q: {e}")
            total_time_ner += (time.time() - t_ner_sub)

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
            "retrieved_docs": retrieved_docs.copy(),
            "response": response_json,
            "extracted_entities": merged_for_gfm,
            "cumulative_facts": cumulative_facts.copy(),
        })

        # =================================================================
        # YÊU CẦU MỚI: Nếu có sub_question và chưa max step thì ép đi tiếp,
        # bỏ qua final_answer ảo do LLM vô tình sinh ra.
        # =================================================================
        if sub_q and step < cfg.test.max_steps:
            logger.info(f"[Step {step}] LLM có yêu cầu sub_question. Bỏ qua final_answer (nếu có) để tiếp tục đào sâu.")
            found_final_answer = None  # Reset để không bị break khỏi vòng lặp
        elif found_final_answer:
            logger.info(f"Đã tìm thấy final_answer ở bước {step}")
            break

        step += 1

        if merged_for_gfm:
            all_discovered_entities.update(e.lower() for e in merged_for_gfm)
            extra_queries: List[str] = []
            if last_missing_entities: extra_queries.append(" ".join(last_missing_entities))
            if sub_q: extra_queries.append(sub_q)

            # Xóa toàn bộ pool cũ trước khi tìm kiếm cho step mới
            global_chunk_pool.clear()
            logger.info(f"[Step-{step}] Đã XÓA SẠCH global_chunk_pool. Bắt đầu tìm kiếm mới hoàn toàn.")

            fetch_and_fuse_into_pool(query_entities=merged_for_gfm, extra_bm25_queries=extra_queries or None, label=f"[Step-{step}]")

            retrieved_docs = rerank_pool([sub_q]) if sub_q else rerank_pool([current_query])

        elif sub_q and (use_bm25 and bm25_searcher):
            logger.info(f"Không có entities, BM25 keyword fallback với sub_q: '{sub_q}'")
            t_bm25_fb = time.time()
            sq_chunks = bm25_searcher.search_standard(sub_q, top_k=cfg.test.top_k)

            # Xóa toàn bộ pool cũ
            global_chunk_pool.clear()
            logger.info(f"[Step-{step}] Đã XÓA SẠCH global_chunk_pool cho sub_q fallback.")

            added = 0
            for c in sq_chunks:
                from gfmrag_hybrid.utils.qa_utils import _make_chunk_key
                cid = _make_chunk_key(c)
                if cid not in global_chunk_pool:
                    c_copy = c.copy()
                    c_copy["bm25_score"] = float(c.get("bm25_score", 0.0))
                    global_chunk_pool[cid] = c_copy
                    added += 1
                else:
                    global_chunk_pool[cid]["bm25_score"] = float(c.get("bm25_score", 0.0))
            logger.info(f"BM25 sub_q fallback thêm {added} chunks")
            total_time_bm25 += (time.time() - t_bm25_fb)
            retrieved_docs = rerank_pool([sub_q])
        else:
            retrieved_docs = rerank_pool([current_query])

    # --- Tổng hợp câu trả lời cuối ---
    # LUÔN gộp toàn bộ sự kiện đã xác nhận để trả lời ĐẦY ĐỦ mọi ý của câu hỏi gốc.
    # (Câu hỏi có thể gồm nhiều phần; final_answer từng bước dễ chỉ trả lời một phần.)
    if cumulative_facts:
        facts_text = json.dumps(cumulative_facts, ensure_ascii=False)
        synth_prompt = (
            "Bạn là trợ lý y khoa. Chỉ dùng các SỰ KIỆN ĐÃ XÁC NHẬN dưới đây, tuyệt đối không bịa thêm.\n"
            f"SỰ KIỆN ĐÃ XÁC NHẬN:\n{facts_text}\n\n"
            f"CÂU HỎI (có thể gồm NHIỀU ý):\n{current_query}\n\n"
            "Yêu cầu: Trả lời ĐẦY ĐỦ từng ý của câu hỏi bằng tiếng Việt, mạch lạc. "
            "Nếu một ý không có dữ liệu trong các sự kiện trên, nêu rõ 'chưa có thông tin' cho ý đó.\n"
            "Trả lời:"
        )
        try:
            t_llm2 = time.time()
            synth_answer = llm.generate_sentence(synth_prompt).strip()
            total_time_llm += (time.time() - t_llm2)
            if synth_answer:
                found_final_answer = synth_answer
        except Exception:
            pass

    final_output = found_final_answer or "Tài liệu hiện tại không đủ thông tin để kết luận."

    if isinstance(final_output, str):
        if final_output.startswith("Vậy câu trả lời là:"): final_output = final_output.replace("Vậy câu trả lời là:", "").strip()
        stripped = final_output.strip()
        if (stripped.startswith("{") and stripped.endswith("}")) or (stripped.startswith("[\"") and stripped.endswith("\"]")):
            try:
                parsed = json.loads(stripped.replace("'", '"'))
                if isinstance(parsed, (dict, list)): final_output = _dict_to_prose(parsed)
            except: pass
    elif isinstance(final_output, (dict, list)): final_output = _dict_to_prose(final_output)

    time_total = time.time() - time_total_start
    timings = {
        "NER": total_time_ner, "GFM": total_time_gfm, "BM25": total_time_bm25,
        "Rerank": total_time_rerank, "LLM": total_time_llm,
        "Xử lý khác": max(0.0, time_total - sum([total_time_ner, total_time_gfm, total_time_bm25, total_time_rerank, total_time_llm])),
        "Tổng thời gian": time_total
    }

    return {
        "response": final_output,
        "retrieved_docs": retrieved_docs[:cfg.test.top_k_chunks],
        "logs": logs,
        "timings": timings
    }