import re
from typing import List, Dict
import logging

from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class DocumentChunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 50, model_name: str = "dangvantuan/vietnamese-embedding", device: str | None = None):
        """
        Lưu ý: chunk_size và overlap được giữ lại ở tham số để tương thích với file .yaml cấu hình cũ của bạn.
        Với Semantic Chunker, độ dài của chunk sẽ do AI tự động quyết định 100% dựa trên mức độ thay đổi ngữ nghĩa.

        device: "cuda" | "cpu" | None. None -> tự nhận diện (cuda nếu có, không thì cpu).
        """
        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:  # noqa: BLE001
                device = "cpu"
        logger.info(f"Khởi tạo Semantic Chunker với mô hình nhúng: {model_name} (device={device})...")

        # 1. Khởi tạo mô hình Embedding (tận dụng GPU nếu có, fallback CPU)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device}
        )

        # 2. Khởi tạo Semantic Chunker
        # Sử dụng 'percentile': Cắt khi độ chênh lệch ngữ nghĩa giữa 2 câu vượt qua 95% mức trung bình
        self.text_splitter = SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type="percentile"
        )
        logger.info("Đã khởi tạo xong Semantic Chunker!")

    def extract_entity_chunks(self, document: Dict, target_entities: List[str]) -> List[Dict]:
        """Cắt văn bản bằng AI (Semantic), sau đó lọc ra các chunk chứa Entities"""
        doc_title = document.get("title", "")
        doc_content = document.get("content", "")

        if not doc_content or not target_entities:
            return []

        try:
            # 1. AI đọc và chia văn bản thành các khối thống nhất về ngữ nghĩa
            raw_chunks = self.text_splitter.split_text(doc_content)
        except Exception as e:
            logger.error(f"Lỗi khi cắt ngữ nghĩa bài {doc_title}: {e}")
            # Fallback an toàn nếu AI bị lỗi: Trả về nguyên bài
            raw_chunks = [doc_content]

        valid_chunks = []

        # 2. Quét qua từng chunk ngữ nghĩa, lọc ra chunk có chứa Entity mục tiêu
        for i, chunk_text in enumerate(raw_chunks):
            found_entities = set()
            for entity in target_entities:
                # Dùng Regex để tìm chính xác từ (Word boundary)
                pattern = r'(?<!\w)' + re.escape(entity) + r'(?!\w)'
                if re.search(pattern, chunk_text, re.IGNORECASE):
                    found_entities.add(entity)

            # Nếu chunk này chứa ít nhất 1 entity mục tiêu -> Giữ lại
            if found_entities:
                start_pos = doc_content.find(chunk_text)
                end_pos = start_pos + len(chunk_text) if start_pos != -1 else -1

                valid_chunks.append({
                    "chunk_id": f"{doc_title}_{i}",
                    "document_title": doc_title,
                    "text": chunk_text.strip(),
                    "entities": list(found_entities),
                    "position": (start_pos, end_pos)
                })

        return valid_chunks

    def chunk_document(self, title: str, content: str, language: str = "vi") -> List[Dict]:
        """Tách toàn bộ một tài liệu thành các chunk ngữ nghĩa (không lọc entity).

        Dùng cho splitter (stage0). Mỗi chunk gồm: chunk_id, document_title, text,
        tokenized_text (tách từ theo ngôn ngữ).
        """
        from gfmrag.utils.text_tokenize import tokenize

        if not content:
            return []
        try:
            raw_chunks = self.text_splitter.split_text(content)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Lỗi khi cắt ngữ nghĩa bài {title}: {e}")
            raw_chunks = [content]

        chunks: List[Dict] = []
        for i, chunk_text in enumerate(raw_chunks):
            text = chunk_text.strip()
            if not text:
                continue
            chunks.append({
                "chunk_id": f"{title}_{i}",
                "document_title": title,
                "text": text,
                "tokenized_text": tokenize(text, language),
            })
        return chunks