import os
import re

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET_SDK"] = "1"

import io
import json
import logging
import sys
from typing import List, Dict, Optional

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import torch
from sentence_transformers import CrossEncoder

# ── Dùng subclass expose entity scores thay vì GFMRetriever gốc ──────────────
from gfmrag_hybrid.gfm.retriever_with_entity_scores import GFMRetrieverWithEntityScores

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

from gfmrag_hybrid.bm25 import (
    BM25Searcher,
    normalize_entities,
    VIETNAMESE_STOPWORDS,
)


# =========================================================================
# FORMAT DICT/LIST TO PROSE HELPER
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
) -> dict:
    step = 1
    current_query = query
    all_thoughts: List[str] = []
    logs = []

    precomputed_path = cfg.get("precomputed_chunks_path", None)
    top_entity_k = int(cfg.test.get("top_entity_k", 30))

    # ── Bước 0: NER vòng đầu ──────────────────────────────────────────────────
    raw_entities = gfmrag_retriever.ner_model(current_query)
    entities = normalize_entities(raw_entities)
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
        if not query_entities:
            return 0

        max_gfm_chunks = int(cfg.test.get("max_gfm_chunks", 20))
        max_bm25_chunks = int(cfg.test.get("max_bm25_chunks", 20))

        # =========================================================
        # NHÁNH 1: TRUY XUẤT GFM
        # =========================================================
        gfm_chunks, entity_scores = retrieve_chunks_with_entity_scores(
            retriever=gfmrag_retriever,
            entities=query_entities,
            top_k=cfg.test.top_k,
            precomputed_path=precomputed_path,
            top_entity_k=top_entity_k,
            max_total_chunks=max_gfm_chunks
        )

        logger.info(
            f"{label} Nhánh GFM: Trả về {len(gfm_chunks)} chunks, "
            f"{len(entity_scores)} entity scores (top: "
            + (f"{entity_scores[0].entity_name}={entity_scores[0].norm_score:.3f}" if entity_scores else "none")
            + ")"
        )

        # =========================================================
        # NHÁNH 2: TRUY XUẤT BM25 (GỘP CHUỖI & GIỚI HẠN max_bm25_chunks)
        # =========================================================
        top_bm25_chunks: List[Dict] = []
        if bm25_searcher:
            # 1. Khởi tạo danh sách từ khóa (từ truy vấn gốc)
            combined_terms = [e for e in query_entities if e.strip()]
            base_set = set(e.lower().strip() for e in combined_terms)

            # 2. Lọc thêm các entities từ GFM (bỏ qua những cái đã có mặt)
            if entity_scores:
                for e in entity_scores:
                    name = e.entity_name if hasattr(e, "entity_name") else e.get("entity_name", "")
                    name_lower = name.lower().strip()
                    score = e.norm_score if hasattr(e, "norm_score") else e.get("norm_score", 0.0)

                    if name_lower and score >= 0.1 and name_lower not in base_set:
                        combined_terms.append(name)
                        base_set.add(name_lower)

            # 3. Thêm các câu hỏi phụ (nếu có)
            if extra_bm25_queries:
                for eq in extra_bm25_queries:
                    eq_lower = eq.lower().strip()
                    if eq_lower and eq_lower not in base_set:
                        combined_terms.append(eq.strip())
                        base_set.add(eq_lower)

            # 4. Gộp thành 1 chuỗi duy nhất
            combined_query_string = " ".join(combined_terms)
            logger.info(f"{label} BM25 Combined Query: '{combined_query_string}'")

            # 5. TÌM KIẾM VÀ ÉP LẤY ĐÚNG SỐ LƯỢNG max_bm25_chunks TỪ CONFIG
            top_bm25_chunks = bm25_searcher.search_standard(
                query=combined_query_string,
                top_k=max_bm25_chunks
            )
            logger.info(
                f"{label} Nhánh BM25: Đã lấy chính xác {len(top_bm25_chunks)} chunks (Giới hạn: {max_bm25_chunks}).")
        # =========================================================
        # BƯỚC 3: ĐẨY HAI NHÁNH VÀO GLOBAL POOL (Xử lý hợp nhất trường hợp trùng lặp)
        # =========================================================
        added = 0
        # 1. Đẩy nhánh GFM trước
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

        # 2. Đẩy nhánh BM25 sau (Điểm bm25_score giờ đã là điểm BM25 gốc)
        for c in top_bm25_chunks:
            from gfmrag_hybrid.utils.qa_utils import _make_chunk_key
            cid = _make_chunk_key(c)
            if cid not in global_chunk_pool:
                global_chunk_pool[cid] = c.copy()
                added += 1
            else:
                global_chunk_pool[cid]["bm25_score"] = c.get("bm25_score", 0.0)

        logger.info(
            f"{label} Bơm thẳng {added} chunks vào pool chờ Cross-Encoder "
            f"(Current Pool size: {len(global_chunk_pool)})"
        )

    # Khởi tạo đa nhịp
    is_multi_hop = len(entities) >= 2
    extra_init = [" ".join(entities), current_query] if is_multi_hop else None
    fetch_and_fuse_into_pool(query_entities=entities, extra_bm25_queries=extra_init, label="[Step-0]")

    if not global_chunk_pool and bm25_searcher:
        logger.warning("[Step-0] Pool empty GFM, fallback BM25 keyword...")
        combined_fallback = " ".join([current_query] + entities)
        fallback = bm25_searcher.search_standard(combined_fallback, top_k=cfg.test.top_k * 2)
        for c in fallback:
            from gfmrag_hybrid.utils.qa_utils import _make_chunk_key
            cid = _make_chunk_key(c)
            if cid not in global_chunk_pool:
                c_copy = c.copy()
                c_copy["bm25_score"] = float(c.get("bm25_score", 0.0))
                global_chunk_pool[cid] = c_copy

    # =========================================================================
    # RERANK POOL
    # =========================================================================
    def rerank_pool(target_queries: List[str]) -> List[Dict]:
        pool_docs = list(global_chunk_pool.values())

        logger.info(f"[Rerank] Reranker. chunks in global_chunk_pool: {len(pool_docs)}")

        if not pool_docs:
            return []

        valid_queries = [q for q in target_queries if q and q.strip()] or [current_query]

        for chunk in pool_docs:
            chunk["max_score"] = -999.0

        for q in valid_queries:
            pairs = [
                [q,
                 f"Tiêu đề: {c.get('document_title', c.get('title', 'Unknown'))} | Nội dung: {c.get('text', c.get('content', ''))}"]
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
                "rrf_doc": chunk.get("rrf_doc", 0.0),
                "rrf_entity": chunk.get("rrf_entity", 0.0),
                "document_norm_score": chunk.get("document_norm_score", 0.0),
                "bm25_score": chunk.get("bm25_score", 0.0),
            })

        return sorted(ranked, key=lambda x: x["score"], reverse=True)

    retrieved_docs = rerank_pool([current_query])

    found_final_answer = None
    cumulative_facts: Dict = {}

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
            if last_missing_entities:
                extra_queries.append(" ".join(last_missing_entities))
            if sub_q:
                extra_queries.append(sub_q)

            # Xóa toàn bộ pool cũ trước khi tìm kiếm cho step mới
            global_chunk_pool.clear()
            logger.info(f"[Step-{step}] Đã XÓA SẠCH global_chunk_pool. Bắt đầu tìm kiếm mới hoàn toàn.")

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
            retrieved_docs = rerank_pool([sub_q])
        else:
            retrieved_docs = rerank_pool([current_query])

    # --- Cơ chế Fallback khi không tìm thấy final answer ---
    if not found_final_answer and cumulative_facts:
        facts_text = json.dumps(cumulative_facts, ensure_ascii=False)
        fallback_prompt = (
            f"Dựa hoàn toàn vào các sự kiện đã xác nhận sau:\n{facts_text}\n\nTrả lời câu hỏi:\n{current_query}\nTrả lời:")
        try:
            fallback_answer = llm.generate_sentence(fallback_prompt).strip()
            if fallback_answer and not any(
                    p in fallback_answer.lower() for p in ["không đủ", "không thể", "không biết"]):
                found_final_answer = fallback_answer
        except:
            pass

    final_retrieved_docs = retrieved_docs

    final_output = found_final_answer or "Tài liệu hiện tại không đủ thông tin để kết luận."

    # Xử lý định dạng đầu ra
    if isinstance(final_output, str):
        if final_output.startswith("Vậy câu trả lời là:"):
            final_output = final_output.replace("Vậy câu trả lời là:", "").strip()
        stripped = final_output.strip()
        if (stripped.startswith("{") and stripped.endswith("}")) or (
                stripped.startswith("[") and stripped.endswith("]")):
            try:
                parsed = json.loads(stripped.replace("'", '"'))
                if isinstance(parsed, (dict, list)):
                    final_output = _dict_to_prose(parsed)
            except:
                pass
    elif isinstance(final_output, (dict, list)):
        final_output = _dict_to_prose(final_output)

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
    config_name="stage3_qa_ircot_inference_chunks_vietnamese_medical",
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