import hashlib
import logging
import time

import torch
import re
from typing import List, Dict
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from gfmrag_hybrid import utils
from gfmrag_hybrid.datasets import QADataset
from gfmrag_hybrid.doc_rankers import BaseDocRanker
from gfmrag_hybrid.kg_construction.entity_linking_model import BaseELModel
from gfmrag_hybrid.kg_construction.ner_model import BaseNERModel
from gfmrag_hybrid.models import GNNRetriever
from gfmrag_hybrid.text_emb_models import BaseTextEmbModel
from gfmrag_hybrid.ultra import query_utils
from gfmrag_hybrid.utils.qa_utils import entities_to_mask, DocumentRetriever

logger = logging.getLogger(__name__)


def dedup_retrieved_docs(docs: list) -> list:
    """
    Lọc trùng lặp tài liệu dựa trên mã băm nội dung (MD5).
    Giữ lại tài liệu có điểm số cao nhất nếu nội dung trùng nhau.
    """
    seen_content: dict[str, int] = {}  # content_hash -> index trong danh sách deduped
    deduped = []

    for doc in docs:
        # Tạo mã băm từ 200 ký tự đầu tiên để định danh nội dung
        content_key = hashlib.md5(
            doc["content"].strip()[:200].encode("utf-8")
        ).hexdigest()

        if content_key not in seen_content:
            seen_content[content_key] = len(deduped)
            deduped.append(doc)
        else:
            # Nếu tài liệu mới trùng nội dung nhưng có điểm norm_score cao hơn thì thay thế
            existing_idx = seen_content[content_key]
            if doc.get("norm_score", 0) > deduped[existing_idx].get("norm_score", 0):
                deduped[existing_idx] = doc

    return deduped


class GFMRetriever:
    """Graph Foundation Model (GFM) Retriever nâng cấp với cơ chế bypass NER và Deduplication."""

    def __init__(
            self,
            qa_data: QADataset,
            text_emb_model: BaseTextEmbModel,
            ner_model: BaseNERModel,
            el_model: BaseELModel,
            graph_retriever: GNNRetriever,
            doc_ranker: BaseDocRanker,
            doc_retriever: DocumentRetriever,
            entities_weight: torch.Tensor | None,
            device: torch.device,
    ) -> None:
        self.qa_data = qa_data
        self.graph = qa_data.kg
        self.text_emb_model = text_emb_model
        self.ner_model = ner_model
        self.el_model = el_model
        self.graph_retriever = graph_retriever
        self.doc_ranker = doc_ranker
        self.doc_retriever = doc_retriever
        self.device = device
        self.num_nodes = self.graph.num_nodes
        self.entities_weight = entities_weight

    @torch.no_grad()
    def retrieve(self, query: str, top_k: int, pre_extracted_entities: List[str] = None) -> list[dict]:
        t0 = time.time()

        # 1. Bấm giờ khâu chuẩn bị (EL + Embedding)
        t_prep_start = time.time()
        graph_retriever_input = self.prepare_input_for_graph_retriever(
            query, pre_extracted_entities=pre_extracted_entities
        )
        graph_retriever_input = query_utils.cuda(graph_retriever_input, device=self.device)
        print(f"Thời gian Prepare (EL + Embs): {time.time() - t_prep_start:.3f}s")

        # 2. Bấm giờ GNN Graph Retriever
        t_graph_start = time.time()
        ent_pred = self.graph_retriever(
            self.graph, graph_retriever_input, entities_weight=self.entities_weight
        )
        print(f"Thời gian GNN Graph Inference: {time.time() - t_graph_start:.3f}s")

        # 3. Bấm giờ Ranker & Extract
        t_rank_start = time.time()
        doc_pred = self.doc_ranker(ent_pred)[0]
        retrieved_docs = self.doc_retriever(doc_pred.cpu(), top_k=top_k)
        deduped_docs = dedup_retrieved_docs(retrieved_docs)
        print(f"Thời gian Rank & Dedup: {time.time() - t_rank_start:.3f}s")

        print(f"Tổng thời gian GFM retrieve: {time.time() - t0:.3f}s")
        return deduped_docs[:top_k]

    def prepare_input_for_graph_retriever(self, query: str, pre_extracted_entities: List[str] = None) -> dict:
        if pre_extracted_entities and len(pre_extracted_entities) > 0:
            mentioned_entities = pre_extracted_entities
        else:
            mentioned_entities = self.ner_model(query)
            if len(mentioned_entities) == 0:
                mentioned_entities = [query]

        # Bấm giờ EL Model
        t_el = time.time()
        linked_entities = self.el_model(mentioned_entities, topk=1)
        print(f"Thời gian EL_Model: {time.time() - t_el:.3f}s")

        entity_ids = [
            self.qa_data.ent2id[ent[0]["entity"]]
            for ent in linked_entities.values()
            if ent[0]["entity"] in self.qa_data.ent2id
        ]
        question_entities_masks = entities_to_mask(entity_ids, self.num_nodes).unsqueeze(0).to(self.device)

        # Bấm giờ Text Embedding
        t_emb = time.time()
        question_embedding = self.text_emb_model.encode([query], is_query=True, show_progress_bar=False)
        print(f"Thời gian Text Encode: {time.time() - t_emb:.3f}s")

        return {
            "question_embeddings": question_embedding,
            "question_entities_masks": question_entities_masks,
        }

    def retrieve_with_chunks(self, query: str, top_k: int = 5, chunk_size: int = 200,
                             pre_extracted_entities: List[str] = None) -> List[Dict]:
        """
        Truy xuất tài liệu và cắt nhỏ thành các chunks chứa thực thể quan trọng.
        """
        # Bước 1: Lấy tài liệu gốc (áp dụng dedup bên trong retrieve)
        retrieved_docs = self.retrieve(query, top_k=top_k, pre_extracted_entities=pre_extracted_entities)

        # Bước 2: Lấy thực thể để làm căn cứ cắt chunk
        query_entities = pre_extracted_entities if pre_extracted_entities else self.ner_model(query)

        # Bước 3: Trích xuất các đoạn văn bản (chunks) liên quan
        from gfmrag_hybrid.chunkers.document_chunker import DocumentChunker
        chunker = DocumentChunker(chunk_size=chunk_size)
        all_chunks = []

        for doc in retrieved_docs:
            # Lấy danh sách thực thể có trong tài liệu này (nếu có map sẵn)
            doc_entities = getattr(self.qa_data, 'doc2entity_map', {}).get(doc["title"], [])

            # Tìm giao điểm thực thể giữa câu hỏi và tài liệu
            target_entities = list(set(query_entities) & set(doc_entities))

            if not target_entities:
                target_entities = query_entities  # Fallback dùng entities câu hỏi

            chunks = chunker.extract_entity_chunks(doc, target_entities)
            all_chunks.extend(chunks)

        return sorted(all_chunks, key=lambda x: x.get("document_score", 0), reverse=True)

    @staticmethod
    def from_config(cfg: DictConfig) -> "GFMRetriever":
        """Khởi tạo GFMRetriever từ file cấu hình Hydra (Đã ép chạy GPU)."""
        graph_retriever, model_config = utils.load_model_from_pretrained(
            cfg.graph_retriever.model_path
        )

        # 1. ÉP CỨNG CHẠY TRÊN CUDA (NẾU CÓ) THAY VÌ DỰA VÀO utils.get_device()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[INFO] GFMRetriever đang khởi tạo trên thiết bị: {device.type.upper()}")

        # 2. ĐƯA GRAPH & KNOWLEDGE GRAPH LÊN GPU
        graph_retriever = graph_retriever.to(device)
        graph_retriever.eval()
        if hasattr(torch, 'compile') and device.type == 'cuda':
            try:
                # Tự động tối ưu hoá các phép toán GNN
                graph_retriever = torch.compile(graph_retriever)
                logger.info("[INFO] Đã bật torch.compile cho GNN Model.")
            except Exception as e:
                logger.warning(f"Không thể compile GNN: {e}")

        qa_data = QADataset(
            **cfg.dataset,
            text_emb_model_cfgs=OmegaConf.create(
                model_config["text_emb_model_config"]
            ),
        )
        qa_data.kg = qa_data.kg.to(device)
        ent2docs = qa_data.ent2docs.to(device)
        if ent2docs.is_sparse and not ent2docs.is_coalesced():
            ent2docs = ent2docs.coalesce()
            logger.info("[INFO] Đã coalesce ma trận ent2docs để tăng tốc sparse.mm")

        # 3. ĐƯA CÁC MODEL XỬ LÝ NGÔN NGỮ (NER, EL, TEXT EMB) LÊN GPU
        ner_model = instantiate(cfg.graph_retriever.ner_model)
        if hasattr(ner_model, 'to'):
            ner_model = ner_model.to(device)

        el_model = instantiate(cfg.graph_retriever.el_model)
        if hasattr(el_model, 'to'):
            el_model = el_model.to(device)
        el_model.index(list(qa_data.ent2id.keys()))

        text_emb_model = instantiate(
            OmegaConf.create(model_config["text_emb_model_config"])
        )
        if hasattr(text_emb_model, 'to'):
            text_emb_model = text_emb_model.to(device)

        # 4. ĐƯA RANKER LÊN GPU
        doc_ranker = instantiate(cfg.graph_retriever.doc_ranker, ent2doc=ent2docs)
        if hasattr(doc_ranker, 'to'):
            doc_ranker = doc_ranker.to(device)

        doc_retriever = utils.DocumentRetriever(qa_data.doc, qa_data.id2doc)

        # 5. ĐƯA TRỌNG SỐ THỰC THỂ (TENSOR) LÊN GPU
        entities_weight = None
        if cfg.graph_retriever.init_entities_weight:
            entities_weight = utils.get_entities_weight(ent2docs)
            if entities_weight is not None:
                entities_weight = entities_weight.to(device)

        return GFMRetriever(
            qa_data=qa_data,
            text_emb_model=text_emb_model,
            ner_model=ner_model,
            el_model=el_model,
            graph_retriever=graph_retriever,
            doc_ranker=doc_ranker,
            doc_retriever=doc_retriever,
            entities_weight=entities_weight,
            device=device,
        )