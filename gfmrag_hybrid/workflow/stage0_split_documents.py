"""Stage 0 — Splitter: tách tài liệu dài thành chunk (đa ngôn ngữ).

Chiều XUÔI của pipeline tiền xử lý: đọc ``raw/dataset_corpus.json`` (toàn văn),
tách mỗi tài liệu thành các chunk ngữ nghĩa (SemanticChunker) rồi ghi:
  - ``raw/dataset_corpus.json``           : corpus mức chunk {chunk_id: text}
  - ``processed/stage1/precomputed_chunks.json`` : {document_title: [chunk,...]}

Chạy thủ công TRƯỚC stage1:
    python -m gfmrag_hybrid.workflow.stage0_split_documents \
        dataset.data_name=vietnamese_medical language=vi
"""

import json
import logging
import os
import shutil

import dotenv
import hydra
from omegaconf import DictConfig, OmegaConf

from gfmrag_hybrid.chunkers.document_chunker import DocumentChunker
from gfmrag_hybrid.kg_construction.chunk_grouper import DEFAULT_EMBED_MODELS

logger = logging.getLogger(__name__)

dotenv.load_dotenv()


def _resolve_model_name(splitter_cfg, language: str) -> str:
    model_name = splitter_cfg.get("model_name")
    if model_name:
        return model_name
    if language not in DEFAULT_EMBED_MODELS:
        raise ValueError(
            f"Không có embedding mặc định cho language={language!r}; "
            f"hãy đặt splitter.model_name."
        )
    return DEFAULT_EMBED_MODELS[language]


def _dump_json(obj: dict, path: str, backup: bool = True) -> None:
    if backup and os.path.exists(path):
        shutil.copyfile(path, path + ".bak")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _to_text(content) -> str:
    if isinstance(content, list):
        return " ".join(content)
    if isinstance(content, dict):
        return content.get("text", content.get("context", str(content)))
    return str(content)


@hydra.main(
    config_path="config", config_name="stage0_split_documents", version_base=None
)
def main(cfg: DictConfig) -> None:
    logger.info(f"Config:\n {OmegaConf.to_yaml(cfg)}")
    language = cfg.get("language", "vi")
    data_root = cfg.dataset.root
    data_name = cfg.dataset.data_name

    corpus_path = os.path.join(data_root, data_name, "raw", "dataset_corpus.json")
    chunks_path = os.path.join(
        data_root, data_name, "processed", "stage1", "precomputed_chunks.json"
    )

    with open(corpus_path, encoding="utf-8") as f:
        raw_corpus = json.load(f)
    logger.info(f"[stage0] Đọc {len(raw_corpus)} tài liệu từ {corpus_path}")

    model_name = _resolve_model_name(cfg.splitter, language)
    chunker = DocumentChunker(
        chunk_size=cfg.splitter.get("chunk_size", 500),
        overlap=cfg.splitter.get("overlap", 50),
        model_name=model_name,
        device=cfg.splitter.get("device"),
    )

    chunk_corpus: dict[str, str] = {}
    precomputed_chunks: dict[str, list[dict]] = {}

    for doc_title, content in raw_corpus.items():
        text = _to_text(content)
        chunks = chunker.chunk_document(doc_title, text, language=language)
        if not chunks:
            continue
        precomputed_chunks[doc_title] = chunks
        for chunk in chunks:
            chunk_corpus[chunk["chunk_id"]] = chunk["text"]

    backup = cfg.splitter.get("backup", True)
    _dump_json(chunk_corpus, corpus_path, backup=backup)
    _dump_json(precomputed_chunks, chunks_path, backup=backup)
    logger.info(
        f"[stage0] Tách {len(raw_corpus)} tài liệu -> {len(chunk_corpus)} chunk. "
        f"Ghi:\n  - {corpus_path}\n  - {chunks_path}"
        + ("  (đã backup *.bak)" if backup else "")
    )


if __name__ == "__main__":
    main()
