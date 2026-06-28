import time
import os
import re
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import torch
from sentence_transformers import CrossEncoder

from gfmrag import GFMRetriever
from gfmrag.utils.qa_utils import retrieve_chunks_with_pre_extracted_entities
from gfmrag.prompt_builder import QAPromptBuilder

# =========================================================================
# BỘ HÀM CHUẨN HÓA THỰC THỂ TIẾNG ANH (Copy 100% từ file gốc của bạn)
# =========================================================================
_STRIP_PREFIXES = [
    "director of", "founder of", "creator of", "author of", "operator of",
    "headquarters of", "capital of", "president of", "owner of", "leader of",
    "history of", "location of", "origin of", "maker of", "head of",
]
_STRIP_SUFFIXES = [
    "status", "location", "history", "capital", "headquarters",
    "founder", "director", "origin", "background", "details",
]


def normalize_entity(entity: str) -> str:
    e = entity.strip()
    e = re.sub(r'^\[(.+)\]$', r'\1', e).strip()
    for prefix in _STRIP_PREFIXES:
        if e.lower().startswith(prefix):
            e = e[len(prefix):].strip()
            break
    words = e.split()
    while words and words[-1].lower() in _STRIP_SUFFIXES:
        words.pop()
    e = " ".join(words).strip()
    used_in_match = re.match(r'^.+?\s+used\s+in\s+(.+)$', e, re.IGNORECASE)
    if used_in_match:
        e = used_in_match.group(1).strip()
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


def sync_time():
    """Hàm đồng bộ GPU để đo thời gian chính xác nhất"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


# Lưu ý: Giữ nguyên tên config 'icrot' giống trong file của bạn để tránh lỗi Hydra
@hydra.main(config_path="config", config_name="stage3_qa_icrot_inference_chunks", version_base=None)
def test_full_pipeline_speed(cfg: DictConfig):
    # Cài đặt API Key — đọc từ biến môi trường (KHÔNG hardcode khóa thật)
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("Chưa set OPENAI_API_KEY. Hãy export OPENAI_API_KEY='sk-...'")

    print("=" * 60)
    print("1. KHỞI TẠO MÔ HÌNH VÀ CÔNG CỤ (MuSiQue Pipeline)")
    print("=" * 60)
    t_init_start = sync_time()

    retriever = GFMRetriever.from_config(cfg)
    llm = instantiate(cfg.llm)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', device=device, model_kwargs={"torch_dtype": torch.float16})
    qa_prompt_builder = QAPromptBuilder(cfg.agent_prompt)

    print(f"[INFO] Tổng thời gian nạp models: {sync_time() - t_init_start:.2f} giây\n")

    print("=" * 60)
    print("2. CHẠY WARM-UP (VƯỢT QUA COLD-START CUDA)")
    print("=" * 60)
    t_warmup = sync_time()
    _ = retrieve_chunks_with_pre_extracted_entities(retriever=retriever, entities=['dummy test'], top_k=2)
    _ = reranker.predict([["dummy query", "dummy doc"]])
    print(f"[INFO] Thời gian Warm-up: {sync_time() - t_warmup:.2f} giây\n")

    # =====================================================================
    # MÔ PHỎNG LUỒNG IRCOT CHÍNH XÁC (STEP 1)
    # =====================================================================
    print("=" * 60)
    print("3. BẮT ĐẦU ĐO ĐẠC LUỒNG INFERENCE (IRCoT - STEP 1)")
    print("=" * 60)

    # Một câu hỏi Multi-hop tiếng Anh đặc trưng cho tập MuSiQue
    query = 'Do Surface Porosity and Pore Size Influence Mechanical Properties and Cellular Response to PEEK?'

    # --- ĐOẠN 1: TRÍCH XUẤT THỰC THỂ (NER + NORMALIZE) ---
    t0 = sync_time()
    raw_entities = retriever.ner_model(query)
    entities = normalize_entities(raw_entities)  # Dùng hàm normalize của MuSiQue
    t1 = sync_time()
    print(f"[1. NER & Normalize] Thời gian trích xuất thực thể: {t1 - t0:.4f} giây")
    print(f"   -> Entities tìm được: {entities}")

    # --- ĐOẠN 2: TRUY XUẤT TÀI LIỆU (GFM + Chunks) ---
    t2 = sync_time()
    initial_chunks = retrieve_chunks_with_pre_extracted_entities(
        retriever=retriever,
        entities=entities,
        top_k=cfg.test.top_k
    )
    t3 = sync_time()
    print(f"[2. Retrieval Phase] Thời gian GFM tìm kiếm và bốc chunks: {t3 - t2:.4f} giây")
    print(f"   -> Số lượng chunks tìm được: {len(initial_chunks)}")

    # --- ĐOẠN 3: XẾP HẠNG LẠI (RERANKER) ---
    t4 = sync_time()
    if initial_chunks:
        # Chuẩn bị dữ liệu cho Reranker giống hệt file MuSiQue
        pairs = [[query, f"Title: {c.get('document_title', 'Unknown')} | Content: {c.get('text', '')}"] for c in
                 initial_chunks]
        scores = reranker.predict(pairs, batch_size=64)

        for i, score in enumerate(scores):
            initial_chunks[i]["rerank_score"] = float(score)
        ranked_docs = sorted(initial_chunks, key=lambda x: x["rerank_score"], reverse=True)
    else:
        ranked_docs = []
    t5 = sync_time()
    print(f"[3. Reranking Phase] Thời gian Cross-Encoder chấm điểm {len(initial_chunks)} chunks: {t5 - t4:.4f} giây")

    # --- ĐOẠN 4: LLM REASONING (PROMPT + API) ---
    t6 = sync_time()

    # BƯỚC FIX LỖI: Map lại key từ document_title/text sang title/content
    formatted_docs_to_llm = []
    for d in ranked_docs[:cfg.test.top_k_chunks]:
        formatted_docs_to_llm.append({
            "title": d.get("document_title", "Unknown"),
            "content": d.get("text", "")
        })

    memory_str = "{}"
    message = qa_prompt_builder.build_input_prompt(
        query,
        formatted_docs_to_llm,  # Dùng list đã map key
        [f"Confirmed facts extracted from previous steps: {memory_str}"]
    )

    print(f"[4A. Prompt Builder] Đã đóng gói xong Prompt (Tốn {sync_time() - t6:.4f} giây). Gửi API tới LLM...")

    t_call_llm = sync_time()
    raw_response = llm.generate_sentence(message)
    t7 = sync_time()

    print(f"[4B. LLM API Call] Thời gian LLM suy luận và trả JSON: {t7 - t_call_llm:.4f} giây")

    print("-" * 60)
    print(f"🔥 TỔNG THỜI GIAN STEP 1 (NER + Retrieval + Rerank + LLM): {t7 - t0:.2f} giây 🔥")
    print("-" * 60)


if __name__ == "__main__":
    test_full_pipeline_speed()