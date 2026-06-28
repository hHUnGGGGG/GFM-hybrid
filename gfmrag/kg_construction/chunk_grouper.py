"""Gom cụm chunk/document theo chủ đề (cùng bệnh/thuốc) thành một tài liệu.

Đây là chiều NGƯỢC của splitter (stage0): lấy các chunk/tài liệu tương quan ngữ
nghĩa, gom thành một tài liệu gộp rồi ghi đè ``dataset_corpus.json`` +
``precomputed_chunks.json``. Được ``stage1_index_dataset`` tự gọi trước khi xây KG.

Đa ngôn ngữ qua tham số ``language`` (vi | en): chọn embedding mặc định và cách
tách từ tương ứng.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from gfmrag.utils.text_tokenize import tokenize

logger = logging.getLogger(__name__)

__all__ = ["GrouperConfig", "ChunkGrouper", "run_chunk_process", "DEFAULT_EMBED_MODELS"]

# Embedding mặc định theo ngôn ngữ (override bằng GrouperConfig.model_name).
DEFAULT_EMBED_MODELS = {
    "vi": "dangvantuan/vietnamese-embedding",
    "en": "BAAI/bge-large-en",
}


@dataclass
class GrouperConfig:
    language: str = "vi"  # vi | en
    model_name: str | None = None  # None -> tự chọn theo language
    distance_threshold: float = 0.075
    linkage: str = "average"  # "average" ổn định với cosine distance
    use_llm_naming: bool = True
    llm_base_url: str = "https://api.yescale.io/v1"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))

    def resolve_model_name(self) -> str:
        if self.model_name:
            return self.model_name
        if self.language not in DEFAULT_EMBED_MODELS:
            raise ValueError(
                f"Không có embedding mặc định cho language={self.language!r}; "
                f"hãy đặt chunk_grouping.model_name."
            )
        return DEFAULT_EMBED_MODELS[self.language]


class ChunkGrouper:
    """Embedding + agglomerative clustering + (tùy chọn) LLM đặt tên cụm."""

    def __init__(self, config: GrouperConfig):
        from sentence_transformers import SentenceTransformer

        self.config = config
        self._embeddings_cache: dict[tuple, np.ndarray] = {}
        model_name = config.resolve_model_name()
        logger.info(f"[ChunkGrouper] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        cache_key = (len(texts), hash(texts[0]) if texts else 0)
        if cache_key in self._embeddings_cache:
            logger.info("[ChunkGrouper] Using cached embeddings")
            return self._embeddings_cache[cache_key]
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=len(texts) > 20,
            normalize_embeddings=True,
        )
        self._embeddings_cache[cache_key] = embeddings
        return embeddings

    def cluster(self, embeddings: np.ndarray):
        distance_matrix = 1.0 - cosine_similarity(embeddings)
        distance_matrix = np.clip(distance_matrix, 0.0, 2.0)
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.config.distance_threshold,
            metric="precomputed",
            linkage=self.config.linkage,
        )
        labels = clustering.fit_predict(distance_matrix)
        return labels, distance_matrix

    def name_clusters_with_llm(self, label_to_texts: dict[int, list[str]]) -> dict[int, str]:
        """Trả về {label: tên cụm}. Lỗi/timeout -> fallback 'Cluster {label}'."""
        from openai import OpenAI

        client = OpenAI(api_key=self.config.llm_api_key, base_url=self.config.llm_base_url)
        names: dict[int, str] = {}
        for label, texts in label_to_texts.items():
            snippets = "\n\n".join(
                f"[Doc {i + 1}]: {t[:400]}..." for i, t in enumerate(texts[:3])
            )
            prompt = (
                "You are a librarian organizing a knowledge base.\n"
                "Given these snippets from the same thematic cluster, provide a "
                "concise cluster name (5-8 words) that can serve as a document title.\n\n"
                f"Documents:\n{snippets}\n\n"
                'Respond in JSON only: {"name": "..."}'
            )
            try:
                resp = client.chat.completions.create(
                    model=self.config.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=120,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                result = json.loads(resp.choices[0].message.content)
                names[label] = (result.get("name") or f"Cluster {label}").strip()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[ChunkGrouper] LLM naming failed for cluster {label}: {e}")
                names[label] = f"Cluster {label}"
        return names

    def group(self, units: list[dict]) -> tuple[list[int], dict[int, str]]:
        """Gom cụm danh sách unit (mỗi unit có khóa 'text').

        Returns:
            (labels, names) — labels theo thứ tự units; names: {label: tên cụm}.
        """
        texts = [u["text"] for u in units]
        logger.info(f"[ChunkGrouper] Embedding {len(texts)} units...")
        embeddings = self.embed(texts)
        logger.info("[ChunkGrouper] Clustering...")
        labels, _ = self.cluster(embeddings)
        n_clusters = len(set(int(x) for x in labels))
        logger.info(f"[ChunkGrouper] Found {n_clusters} clusters from {len(texts)} units")

        if self.config.use_llm_naming:
            label_to_texts: dict[int, list[str]] = {}
            for unit, label in zip(units, labels):
                label_to_texts.setdefault(int(label), []).append(unit["text"])
            logger.info("[ChunkGrouper] Naming clusters with LLM...")
            names = self.name_clusters_with_llm(label_to_texts)
        else:
            names = {int(label): f"Cluster {int(label)}" for label in set(labels)}
        return [int(x) for x in labels], names


# ─────────────────────────────────────────────
# Pure helpers (tách riêng để test không cần model)
# ─────────────────────────────────────────────

def _dedupe_name(name: str, used: set[str]) -> str:
    """Tránh trùng tên document gộp."""
    candidate = name
    i = 1
    while candidate in used:
        i += 1
        candidate = f"{name} ({i})"
    used.add(candidate)
    return candidate


def build_grouped_outputs(
    units: list[dict],
    labels: list[int],
    names: dict[int, str],
    language: str = "vi",
) -> tuple[dict[str, str], dict[str, list[dict]]]:
    """Gộp các unit cùng cụm thành 1 document.

    Mỗi unit: ``{"key": str, "text": str, "chunks": list[chunk dict]}``.
    Trả về (dataset_corpus, precomputed_chunks) ở mức document.
    """
    clusters: dict[int, list[dict]] = {}
    for unit, label in zip(units, labels):
        clusters.setdefault(int(label), []).append(unit)

    dataset_corpus: dict[str, str] = {}
    precomputed_chunks: dict[str, list[dict]] = {}
    used_names: set[str] = set()

    for label in sorted(clusters):
        members = clusters[label]
        title = _dedupe_name(names.get(label, f"Cluster {label}"), used_names)

        merged_text = "\n\n".join(m["text"] for m in members)
        dataset_corpus[title] = merged_text

        merged_chunks: list[dict] = []
        idx = 0
        for member in members:
            for chunk in member["chunks"]:
                text = chunk.get("text", "")
                tokenized = chunk.get("tokenized_text") or tokenize(text, language)
                merged_chunks.append(
                    {
                        "chunk_id": f"{title}_{idx}",
                        "document_title": title,
                        "text": text,
                        "tokenized_text": tokenized,
                    }
                )
                idx += 1
        precomputed_chunks[title] = merged_chunks

    return dataset_corpus, precomputed_chunks


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _dump_json(obj: dict, path: str, backup: bool = True) -> None:
    if backup and os.path.exists(path):
        shutil.copyfile(path, path + ".bak")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _build_units(
    granularity: str,
    corpus: dict[str, str],
    chunks: dict[str, list[dict]],
) -> list[dict]:
    """Tạo units để gom cụm theo granularity."""
    if granularity == "chunk":
        units = []
        for doc_title, chunk_list in chunks.items():
            for chunk in chunk_list:
                units.append(
                    {
                        "key": chunk.get("chunk_id", doc_title),
                        "text": chunk.get("text", ""),
                        "chunks": [chunk],
                    }
                )
        return units
    if granularity == "document":
        units = []
        for doc_title, text in corpus.items():
            member_chunks = chunks.get(
                doc_title,
                [
                    {
                        "chunk_id": f"{doc_title}_0",
                        "document_title": doc_title,
                        "text": text,
                    }
                ],
            )
            units.append({"key": doc_title, "text": text, "chunks": member_chunks})
        return units
    raise ValueError(f"granularity không hợp lệ: {granularity!r} (chunk | document)")


def run_chunk_process(cfg) -> None:
    """Đọc config stage1, gom cụm, GHI ĐÈ corpus + precomputed_chunks.

    Đọc:
        cfg.dataset.root, cfg.dataset.data_name
        cfg.language (vi | en)
        cfg.chunk_grouping.{granularity, model_name, distance_threshold, linkage,
                            use_llm_naming, llm_base_url, llm_model, backup}
    """
    gcfg = cfg.chunk_grouping
    language = cfg.get("language", "vi")
    data_root = cfg.dataset.root
    data_name = cfg.dataset.data_name

    corpus_path = os.path.join(data_root, data_name, "raw", "dataset_corpus.json")
    chunks_path = os.path.join(
        data_root, data_name, "processed", "stage1", "precomputed_chunks.json"
    )

    granularity = gcfg.get("granularity", "chunk")
    corpus = _load_json(corpus_path) if os.path.exists(corpus_path) else {}
    chunks = _load_json(chunks_path) if os.path.exists(chunks_path) else {}

    if granularity == "chunk" and not chunks:
        raise FileNotFoundError(
            f"Không tìm thấy precomputed_chunks tại {chunks_path}. "
            f"Hãy chạy splitter (stage0) trước khi gom cụm mức chunk."
        )
    if granularity == "document" and not corpus:
        raise FileNotFoundError(f"Không tìm thấy dataset_corpus tại {corpus_path}.")

    units = _build_units(granularity, corpus, chunks)
    logger.info(
        f"[run_chunk_process] granularity={granularity}, language={language}, "
        f"units={len(units)}"
    )

    config = GrouperConfig(
        language=language,
        model_name=gcfg.get("model_name"),
        distance_threshold=gcfg.get("distance_threshold", 0.075),
        linkage=gcfg.get("linkage", "average"),
        use_llm_naming=gcfg.get("use_llm_naming", True),
        llm_base_url=gcfg.get("llm_base_url", "https://api.yescale.io/v1"),
        llm_model=gcfg.get("llm_model", "gpt-4o-mini"),
    )
    grouper = ChunkGrouper(config)
    labels, names = grouper.group(units)

    new_corpus, new_chunks = build_grouped_outputs(units, labels, names, language)

    backup = gcfg.get("backup", True)
    _dump_json(new_corpus, corpus_path, backup=backup)
    _dump_json(new_chunks, chunks_path, backup=backup)
    logger.info(
        f"[run_chunk_process] Đã gộp {len(units)} units -> {len(new_corpus)} documents. "
        f"Ghi đè:\n  - {corpus_path}\n  - {chunks_path}"
        + ("  (đã backup *.bak)" if backup else "")
    )
