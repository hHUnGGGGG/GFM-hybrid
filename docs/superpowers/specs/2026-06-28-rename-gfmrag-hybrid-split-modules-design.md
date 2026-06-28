# Đổi tên `gfmrag` → `gfmrag_hybrid` và tách hai sub-package `bm25/` + `gfm/`

**Ngày:** 2026-06-28
**Trạng thái:** Đã duyệt thiết kế, chờ viết plan

## 1. Mục tiêu

1. Đổi tên gói Python `gfmrag` thành `gfmrag_hybrid` (cả thư mục import lẫn
   distribution name trong `pyproject.toml`).
2. Trong gói mới, tổ chức lại thành hai sub-package có ranh giới rõ ràng:
   - `gfmrag_hybrid/bm25/` — nhánh truy hồi từ vựng (phần đóng góp hybrid).
   - `gfmrag_hybrid/gfm/` — nhánh truy hồi đồ thị (GFM-RAG retriever).

**Ba mục tiêu người dùng chốt:** code sạch / dễ đọc; tách rõ phần đóng góp
hybrid khỏi GFM-RAG upstream; chạy/thử nghiệm độc lập từng nhánh.

**Hướng đã chọn:** Hướng A (tách ở mức retriever + dọn trùng lặp), KHÔNG di dời
toàn bộ hạ tầng upstream.

## 2. Bối cảnh hiện trạng

- Gói `gfmrag` gồm 75 file `.py`. Chuỗi `gfmrag` xuất hiện ~433 lần trên 42 file
  `.py`, 37 file config `.yaml/.yml`, 30 file `.md`. Không có biến thể
  `gfm-rag` / `gfm_rag` trong code (chỉ một tham chiếu `gfm-rag/` trong README
  cho đường dẫn `.env`, KHÔNG đụng tới).
- **Trùng lặp lớn:** `BM25Searcher`, `normalize_entity` / `normalize_entities`,
  `VIETNAMESE_STOPWORDS`, và `agent_reasoning_with_reranker` bị chép gần như y hệt
  trong 5 file:
  - `gfmrag/workflow/core_engine.py` (bản dùng làm chuẩn — không có `@hydra.main`)
  - `gfmrag/workflow/stage3_qa_ircot_inference_chunks.py` (dùng `ENGLISH_STOPWORDS`)
  - `gfmrag/workflow/stage3_qa_ircot_inference_chunks_vietnamese_medical.py`
  - `gfmrag/workflow/update.py`
  - `gfmrag/workflow/test.py` (chỉ có `normalize_entity(s)`)
- Nhánh GFM hiện ở: `gfmrag/gfmrag_retriever.py` (class `GFMRetriever`,
  hàm `dedup_retrieved_docs`) và `gfmrag/gfmrag_retriever_with_entity_scores.py`
  (class `GFMRetrieverWithEntityScores`, dataclass `EntityScore`).
- `BM25Searcher` nhận `stopwords` qua tham số constructor → class là chung,
  chỉ hằng số stopwords (vi/en) khác nhau theo từng script gọi.
- `app.py` import bằng `from core_engine import ...` (không qua package, chạy với
  cwd = thư mục `workflow`).
- `utils/qa_utils.py` import `EntityScore` từ retriever-with-entity-scores và cung
  cấp `retrieve_chunks_with_entity_scores` + `_make_chunk_key` cho engine.

## 3. Phạm vi đổi tên (Phần 1)

### 3.1 Quy tắc thay thế
- `git mv gfmrag gfmrag_hybrid` để giữ lịch sử git của từng file.
- Thay theo **ranh giới từ** `\bgfmrag\b` → `gfmrag_hybrid` trên các đuôi:
  `.py .yaml .yml .toml .md .ini .cfg .txt` (loại trừ thư mục `.git/`).
- Ranh giới từ là bắt buộc: `gfmrag.workflow` / `gfmrag/ultra` / `"gfmrag"` PHẢI
  đổi, nhưng tiền tố tên file con `gfmrag_retriever` (ký tự kế là `_`, vẫn là
  word char) PHẢI giữ nguyên. Regex `\bgfmrag\b` thỏa cả hai điều này.

### 3.2 Các điểm cụ thể
- `pyproject.toml`: `name = "gfmrag"` → `"gfmrag_hybrid"`;
  `packages = [{ include = "gfmrag" }]` → `include = "gfmrag_hybrid"`;
  `exclude = ["gfmrag/ultra"]` → `["gfmrag_hybrid/ultra"]`.
- Toàn bộ import `from gfmrag...` / `import gfmrag` / Hydra `_target_: gfmrag....`
  trong config, lệnh `python -m gfmrag.workflow...` trong README/docs.
- Sau bước này: `pip uninstall gfmrag` rồi `pip install -e .` lại.

### 3.3 Kiểm chứng bước rename
- `python -c "import gfmrag_hybrid"` thành công.
- `grep -rn "\bgfmrag\b"` (loại `.git`, loại các tên file con như
  `gfmrag_retriever`) trả về rỗng cho các tham chiếu package.

## 4. Tách module (Phần 2 — Hướng A)

### 4.1 Sub-package `gfmrag_hybrid/bm25/`
```
bm25/
├── __init__.py     # export BM25Searcher, normalize_entity, normalize_entities,
│                   #        VIETNAMESE_STOPWORDS, ENGLISH_STOPWORDS
├── searcher.py     # class BM25Searcher (bản chuẩn duy nhất)
├── normalize.py    # normalize_entity, normalize_entities
└── stopwords.py    # VIETNAMESE_STOPWORDS, ENGLISH_STOPWORDS
```
- Trích bản chuẩn của `BM25Searcher`, `normalize_*`, và hằng số stopwords từ
  `core_engine.py` + `stage3_qa_ircot_inference_chunks.py` (cho `ENGLISH_STOPWORDS`).

**Bước kiểm chứng bắt buộc trước khi hợp nhất:** diff từng bản `BM25Searcher`
(và `normalize_*`) ở cả 5 file so với bản chuẩn. Nếu phát hiện drift về **hành vi**
(không phải chỉ stopwords hay khoảng trắng), DỪNG và báo cho người dùng chọn — KHÔNG
tự ý lấy một bản. Nếu chỉ khác stopword constant → an toàn (đã tham số hóa).

- 5 file chép trùng: xóa khối định nghĩa nội tuyến (`BM25Searcher`,
  `normalize_*`, `*_STOPWORDS`), thay bằng `from gfmrag_hybrid.bm25 import ...`.
  `update.py` và `test.py` xử lý như file thường (cập nhật import, không xóa file).

### 4.2 Sub-package `gfmrag_hybrid/gfm/`
```
gfm/
├── __init__.py                       # export GFMRetriever, GFMRetrieverWithEntityScores,
│                                     #        EntityScore, dedup_retrieved_docs
├── retriever.py                      # ← git mv gfmrag_retriever.py
└── retriever_with_entity_scores.py   # ← git mv gfmrag_retriever_with_entity_scores.py
```
- Cập nhật import nội bộ trong file đã chuyển:
  - `retriever_with_entity_scores.py`: `from gfmrag import GFMRetriever` →
    `from gfmrag_hybrid.gfm import GFMRetriever`;
    `from gfmrag.gfmrag_retriever import dedup_retrieved_docs` →
    `from gfmrag_hybrid.gfm.retriever import dedup_retrieved_docs`.
- Cập nhật mọi nơi import retriever sang đường dẫn `gfm/` mới:
  `utils/qa_utils.py` (`EntityScore`), `workflow/app.py`,
  `workflow/core_engine.py`, các script stage3 (`stage3_qa_ircot_inference.py`,
  `..._chunks.py`, `..._chunks_vietnamese_medical.py`,
  `..._vietnamese_medical.py`), `workflow/update.py`, `workflow/test.py`.

### 4.3 Engine hợp nhất và `__init__` top-level
- `agent_reasoning_with_reranker` và `_dict_to_prose` Ở LẠI
  `workflow/core_engine.py` (engine dùng cả hai nhánh — không thuộc riêng module
  nào). Đổi import của nó sang `gfmrag_hybrid.bm25` và `gfmrag_hybrid.gfm`.
- `gfmrag_hybrid/__init__.py`: re-export `GFMRetriever` (từ `gfm`) và `KGIndexer`
  như cũ → `from gfmrag_hybrid import GFMRetriever` tiếp tục chạy.
- `app.py`: đổi import BM25 (`BM25Searcher`, `VIETNAMESE_STOPWORDS`) sang
  `gfmrag_hybrid.bm25`; engine (`agent_reasoning_with_reranker`) vẫn từ
  `core_engine`.

## 5. Thứ tự thực thi

1. Tạo nhánh feature riêng cho công việc này.
2. **Rename:** `git mv` thư mục + thay `\bgfmrag\b`. Smoke:
   `python -c "import gfmrag_hybrid"`. `pip install -e .` lại.
3. **bm25/:** tạo sub-package, chạy diff-check 5 bản, dọn trùng lặp.
4. **gfm/:** `git mv` hai retriever vào `gfm/`, sửa import nội bộ + mọi importer.
5. **Engine & init:** cập nhật `core_engine.py`, `__init__.py`, `app.py`.
6. **Kiểm chứng:** `pytest`, `mypy`, `pre-commit`, và smoke-test import của
   `core_engine` + từng script stage3 + `app.py`.

## 6. Ngoài phạm vi (YAGNI)

- KHÔNG di dời hạ tầng upstream (`ultra/`, `models.py`, `losses.py`, `datasets/`,
  `kg_construction/`, `llms/`, `text_emb_models/`, `doc_rankers.py`,
  `kg_indexer.py`, `evaluation/`).
- KHÔNG tạo shim/alias tương thích ngược cho đường dẫn module cũ.
- KHÔNG xóa file (kể cả file nháp `update.py` / `test.py`).
- KHÔNG đụng tới tham chiếu `gfm-rag/` (có dấu gạch nối) trong README.

## 7. Tiêu chí hoàn thành

- `import gfmrag_hybrid` và `from gfmrag_hybrid import GFMRetriever` chạy được.
- `BM25Searcher`, `normalize_*`, stopwords chỉ còn định nghĩa MỘT lần (trong
  `bm25/`); 5 file cũ import lại từ đó.
- Hai retriever nằm trong `gfmrag_hybrid/gfm/`; mọi importer trỏ đúng.
- `pytest`, `mypy`, `pre-commit` pass (hoặc giữ nguyên trạng thái pass/fail như
  trước khi thay đổi, không phát sinh lỗi mới do refactor).
- Không còn tham chiếu package `gfmrag` (dạng `\bgfmrag\b`) ngoài tên file con.
