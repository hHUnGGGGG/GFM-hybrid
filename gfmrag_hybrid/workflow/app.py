"""
app.py  —  Chatbot Y Tế Thông Minh (IRCoT Engine)
=================================================
Chạy:
    streamlit run app.py
"""

import os
import pandas as pd
import torch
import streamlit as st
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from sentence_transformers import CrossEncoder

from core_engine import agent_reasoning_with_reranker
from gfmrag_hybrid.bm25 import BM25Searcher, VIETNAMESE_STOPWORDS
from gfmrag_hybrid.gfm.retriever_with_entity_scores import GFMRetrieverWithEntityScores
from gfmrag_hybrid.prompt_builder import QAPromptBuilder

# ============================================================
# CẤU HÌNH TRANG
# ============================================================
st.set_page_config(
    page_title="Chatbot Y Tế IRCoT",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS TÙY CHỈNH
# ============================================================
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] { background: #0f172a; }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    /* Badge nguồn tài liệu */
    .source-badge {
        display: inline-block;
        background: rgba(6,182,212,0.15);
        border: 1px solid rgba(6,182,212,0.4);
        color: #06b6d4;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 12px;
        margin: 3px 4px 3px 0;
    }

    /* Score pill */
    .score-pill {
        background: rgba(16,185,129,0.15);
        border: 1px solid rgba(16,185,129,0.3);
        color: #10b981;
        border-radius: 8px;
        padding: 1px 8px;
        font-size: 11px;
    }

    /* Step header */
    .step-header {
        background: linear-gradient(90deg, rgba(6,182,212,0.15), transparent);
        border-left: 3px solid #06b6d4;
        padding: 6px 12px;
        border-radius: 4px;
        margin: 8px 0 4px;
        font-weight: 600;
        color: #0ea5e9;
    }

    /* Entity chip */
    .entity-chip {
        display: inline-block;
        background: rgba(99,102,241,0.15);
        border: 1px solid rgba(99,102,241,0.35);
        color: #818cf8;
        border-radius: 10px;
        padding: 2px 9px;
        font-size: 12px;
        margin: 2px;
    }

    /* Facts card */
    .inv-card {
        background: rgba(245,158,11,0.08);
        border: 1px solid rgba(245,158,11,0.25);
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 13px;
        color: #fbbf24;
        margin: 6px 0;
    }

    /* Disclaimer */
    .disclaimer {
        background: rgba(239,68,68,0.08);
        border: 1px solid rgba(239,68,68,0.25);
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 13px;
        color: #fca5a5;
    }

    /* Timings Badge */
    .time-pill {
        display: inline-block;
        background: rgba(148, 163, 184, 0.1);
        border: 1px solid rgba(148, 163, 184, 0.4);
        color: #94a3b8;
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 12px;
        margin: 6px 6px 0 0;
    }
    .time-pill-total {
        background: rgba(234, 179, 8, 0.15);
        border-color: rgba(234, 179, 8, 0.5);
        color: #eab308;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# TẢI MODEL (CACHE — CHỈ CHẠY 1 LẦN)
# ============================================================
@st.cache_resource(show_spinner="⚙️ Đang khởi tạo AI Engine và Cơ sở dữ liệu Y khoa...")
def load_system():
    os.environ["OPENAI_API_KEY"] = ""
    if not GlobalHydra.instance().is_initialized():
        initialize(version_base=None, config_path="config")
    cfg = compose(config_name="stage3_qa_ircot_inference_chunks_vietnamese_medical")

    # Dùng class mới khai báo hỗ trợ Entity Scores
    gfmrag_retriever = GFMRetrieverWithEntityScores.from_config(cfg)
    llm = instantiate(cfg.llm)
    prompt_builder = QAPromptBuilder(cfg.agent_prompt)

    bm25_searcher = None
    precomputed_path = cfg.get("precomputed_chunks_path", None)
    if precomputed_path and os.path.exists(precomputed_path):
        bm25_searcher = BM25Searcher(precomputed_path, VIETNAMESE_STOPWORDS)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device, model_kwargs={"torch_dtype": torch.float16})

    return cfg, gfmrag_retriever, reranker, llm, prompt_builder, bm25_searcher


cfg, gfmrag_retriever, reranker, llm, prompt_builder, bm25_searcher = load_system()


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## ⚕️ MedBot IRCoT")
    st.markdown("---")

    rag_method = st.radio(
        "🧠 Chọn phương pháp RAG:",
        options=["Hybrid RAG (GFM + BM25)", "Chỉ dùng GFM (Không BM25)"],
        index=0
    )
    use_bm25 = (rag_method == "Hybrid RAG (GFM + BM25)")

    st.markdown("### 🔧 Thông số Engine")
    st.info(
        f"- **Max IRCoT steps:** {cfg.test.max_steps}\n"
        f"- **Top-K retrieve:** {cfg.test.top_k}\n"
        f"- **Chunks gửi LLM:** {cfg.test.top_k_chunks}\n"
        f"- **BM25:** {'✅ Bật' if use_bm25 and bm25_searcher else '❌ Tắt'}\n"
        f"- **Max GFM Chunks:** {cfg.test.get('max_gfm_chunks', 20)}\n"
        f"- **Max BM25 Chunks:** {cfg.test.get('max_bm25_chunks', 20)}\n"
        f"- **Device:** {'🟢 CUDA' if torch.cuda.is_available() else '🔵 CPU'}"
    )

    st.markdown("### 💡 Câu hỏi mẫu")
    sample_questions = [
        "Những loại thuốc nào khi sử dụng chung với Asevictoria Mediplantex có thể làm giảm tác dụng tránh thai của levonorgestrel do tác động cảm ứng enzyme gan?",
        "Đang uống thuốc Coldko trị cảm cúm mà lỡ uống rượu say thì gan và thần kinh có sao không?",
        "Tại sao phụ nữ trên 35 tuổi hút thuốc không nên dùng Belara?",
        "Đối với trường hợp người lớn và trẻ em trên 12 tuổi để giảm đau thì liều dùng Aspirin MKP 81 như thế nào?",
        "bị cảm cúm thì nên ăn uống như thế nào, uống thuốc Acepron có được không?",
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True, key=f"sample_{q[:20]}"):
            st.session_state["prefill_question"] = q

    st.markdown("---")
    if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

    st.markdown("""
    <div class="disclaimer">
    ⚠️ <strong>Lưu ý:</strong> Thông tin chỉ mang tính tham khảo, không thay thế ý kiến bác sĩ.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# HEADER CHÍNH
# ============================================================
st.markdown("# 🩺 Chatbot Tư Vấn Y Tế")
st.markdown("Hệ thống **IRCoT** tự động tra cứu tài liệu y khoa và suy luận đa bước để trả lời câu hỏi.")
st.divider()


# ============================================================
# HÀM HIỂN THỊ LOG SUY LUẬN
# ============================================================
def render_reasoning_log(logs: list):
    if not logs: return
    with st.expander("🔍 Xem chi tiết quá trình tra cứu & suy luận IRCoT", expanded=False):
        total_steps = len(logs)
        st.markdown(f"**Tổng số bước IRCoT:** `{total_steps}`")
        st.divider()

        for log in logs:
            step_num = log["step"]
            st.markdown(f'<div class="step-header">⚙️ Bước {step_num} / {total_steps}</div>', unsafe_allow_html=True)
            resp = log.get("response", {})
            col1, col2 = st.columns([1, 1])

            with col1:
                if resp.get("reasoning"): st.markdown(f"💭 **Suy luận:** {resp['reasoning']}")
                if resp.get("sub_question"): st.info(f"🤔 **Câu hỏi phụ tự sinh:** {resp['sub_question']}")
                if log.get("extracted_entities"):
                    chips = " ".join(f'<span class="entity-chip">{e}</span>' for e in log["extracted_entities"])
                    st.markdown(f"🏷️ **Thực thể tìm kiếm:**<br>{chips}", unsafe_allow_html=True)

            with col2:
                if log.get("cumulative_facts"):
                    facts_text = "<br>".join(f"<b>{k}:</b> {v}" for k, v in log["cumulative_facts"].items())
                    st.markdown(f'<div class="inv-card">📋 <b>Sự kiện tích lũy:</b><br>{facts_text}</div>',
                                unsafe_allow_html=True)

            docs = log.get("retrieved_docs", [])
            if docs:
                st.markdown(f"📚 **Trạng thái Global Pool sau Rerank (Tổng: {len(docs)} chunks):**")

                df_data = []
                for d in docs[:20]:
                    # Tính tổng điểm GFM (Dự phòng nếu core_engine không gom sẵn)
                    gfm_total = d.get("document_norm_score", d.get("rrf_doc", 0.0) + d.get("rrf_entity", 0.0))

                    df_data.append({
                        "Tiêu đề": d.get("title", "Không rõ"),
                        "CE Rerank": d.get("score", 0.0),             # <--- Điểm của Cross-Encoder
                        "RRF Doc": d.get("rrf_doc", 0.0),
                        "RRF Entity": d.get("rrf_entity", 0.0),
                        "GFM Score": gfm_total,
                        "BM25 Score": d.get("bm25_score", 0.0),       # <--- Điểm nhánh BM25
                        "Nội dung (trích lược)": d.get("content", "")[:80] + "..."
                    })

                df = pd.DataFrame(df_data)

                # Sử dụng st.dataframe để hiển thị bảng
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "CE Rerank": st.column_config.NumberColumn(format="%.4f"),
                        "RRF Doc": st.column_config.NumberColumn(format="%.4f"),
                        "RRF Entity": st.column_config.NumberColumn(format="%.4f"),
                        "GFM Score": st.column_config.NumberColumn(format="%.4f"),
                        "BM25 Score": st.column_config.NumberColumn(format="%.4f"),
                    }
                )

            if resp.get("final_answer"):
                st.success(f"✅ **Câu trả lời tìm thấy ở bước {step_num}:** {resp['final_answer']}")
            st.divider()

# ============================================================
# QUẢN LÝ LỊCH SỬ CHAT
# ============================================================
if "messages" not in st.session_state:
    st.session_state["messages"] = [{
        "role": "assistant",
        "content": "Xin chào! Tôi là Trợ lý Y tế IRCoT 🩺\nHãy chọn câu hỏi mẫu ở sidebar hoặc nhập câu hỏi của bạn bên dưới!",
        "logs": None, "retrieved_docs": None, "timings": None
    }]

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("retrieved_docs"):
            badges = " ".join(f'<span class="source-badge">📄 {d.get("title","?")}</span>' for d in msg["retrieved_docs"][:5])
            st.markdown(f"**Nguồn tài liệu:** {badges}", unsafe_allow_html=True)

        if msg.get("timings"):
            timings = msg["timings"]
            time_html = "".join([f'<span class="time-pill">⏱️ {k}: <b>{v:.2f}s</b></span>' for k, v in timings.items() if k != "Tổng thời gian"])
            total_time = timings.get("Tổng thời gian", 0)
            time_html += f'<span class="time-pill time-pill-total">⏳ Tổng: <b>{total_time:.2f}s</b></span>'
            st.markdown(f"<div>{time_html}</div><br>", unsafe_allow_html=True)

        if msg.get("logs"): render_reasoning_log(msg["logs"])

# ============================================================
# XỬ LÝ INPUT
# ============================================================
prefill = st.session_state.pop("prefill_question", None)
prompt = st.chat_input("Nhập câu hỏi y tế...")
question = prefill or prompt

if question:
    st.session_state["messages"].append({"role": "user", "content": question, "logs": None, "retrieved_docs": None, "timings": None})
    with st.chat_message("user"): st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("🧠 Đang tra cứu và suy luận (IRCoT)..."):
            try:
                result_dict = agent_reasoning_with_reranker(
                    cfg=cfg, gfmrag_retriever=gfmrag_retriever, reranker=reranker,
                    llm=llm, qa_prompt_builder=prompt_builder, query=question, bm25_searcher=bm25_searcher,
                    use_bm25=use_bm25
                )

                final_response = result_dict.get("response")
                logs = result_dict.get("logs", [])
                retrieved_docs = result_dict.get("retrieved_docs", [])
                timings = result_dict.get("timings", {})

                if isinstance(final_response, dict): final_response = "\n".join(f"- **{k}:** {v}" for k, v in final_response.items())
                elif isinstance(final_response, list): final_response = "\n".join(f"- {item}" for item in final_response)

                if not final_response: final_response = "Không tìm thấy đủ tài liệu y khoa để kết luận."

                st.markdown(final_response)

                if retrieved_docs:
                    badges = " ".join(f'<span class="source-badge">📄 {d.get("title","?")}</span>' for d in retrieved_docs[:5])
                    st.markdown(f"**Nguồn tài liệu:** {badges}", unsafe_allow_html=True)

                if timings:
                    time_html = "".join([f'<span class="time-pill">⏱️ {k}: <b>{v:.2f}s</b></span>' for k, v in timings.items() if k != "Tổng thời gian"])
                    total_time = timings.get("Tổng thời gian", 0)
                    time_html += f'<span class="time-pill time-pill-total">⏳ Tổng: <b>{total_time:.2f}s</b></span>'
                    st.markdown(f"<div>{time_html}</div><br>", unsafe_allow_html=True)

                render_reasoning_log(logs)

                st.session_state["messages"].append({
                    "role": "assistant", "content": final_response, "logs": logs, "retrieved_docs": retrieved_docs, "timings": timings
                })

            except Exception as e:
                st.error(f"❌ Lỗi: {str(e)}")