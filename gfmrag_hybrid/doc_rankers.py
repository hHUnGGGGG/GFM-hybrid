import time
from abc import ABC, abstractmethod

import torch


class BaseDocRanker(ABC):
    """
    Abstract class for document ranker

    Args:
        ent2doc (torch.Tensor): Mapping from entity to document
    """

    def __init__(self, ent2doc: torch.Tensor) -> None:
        self.ent2doc = ent2doc

    @abstractmethod
    def __call__(self, ent_pred: torch.Tensor) -> torch.Tensor:
        pass


class SimpleRanker(BaseDocRanker):
    def __call__(self, ent_pred: torch.Tensor) -> torch.Tensor:
        """
        Rank documents based on entity prediction

        Args:
            ent_pred (torch.Tensor): Entity prediction, shape (batch_size, n_entities)

        Returns:
            torch.Tensor: Document ranks, shape (batch_size, n_docs)
        """
        doc_pred = torch.sparse.mm(ent_pred, self.ent2doc)
        return doc_pred


class IDFWeightedRanker(BaseDocRanker):
    """
    Rank documents based on entity prediction with IDF weighting
    """

    def __init__(self, ent2doc: torch.Tensor) -> None:
        super().__init__(ent2doc)
        frequency = torch.sparse.sum(ent2doc, dim=-1).to_dense()
        self.idf_weight = 1 / frequency
        self.idf_weight[frequency == 0] = 0

    def __call__(self, ent_pred: torch.Tensor) -> torch.Tensor:
        """
        Rank documents based on entity prediction with IDF weighting

        Args:
            ent_pred (torch.Tensor): Entity prediction, shape (batch_size, n_entities)

        Returns:
            torch.Tensor: Document ranks, shape (batch_size, n_docs)
        """
        doc_pred = torch.sparse.mm(
            ent_pred * self.idf_weight.unsqueeze(0), self.ent2doc
        )
        return doc_pred


class TopKRanker(BaseDocRanker):
    def __init__(self, ent2doc: torch.Tensor, top_k: int) -> None:
        super().__init__(ent2doc)
        self.top_k = top_k

    def __call__(self, ent_pred: torch.Tensor) -> torch.Tensor:
        """
        Rank documents based on top-k entity prediction

        Args:
            ent_pred (torch.Tensor): Entity prediction, shape (batch_size, n_entities)

        Returns:
            torch.Tensor: Document ranks, shape (batch_size, n_docs)
        """
        top_k_ent_pred = torch.topk(ent_pred, self.top_k, dim=-1)
        masked_ent_pred = torch.zeros_like(ent_pred, device=ent_pred.device)
        masked_ent_pred.scatter_(1, top_k_ent_pred.indices, 1)
        doc_pred = torch.sparse.mm(masked_ent_pred, self.ent2doc)
        return doc_pred

class IDFWeightedTopKRanker(BaseDocRanker):
    def __init__(self, ent2doc: torch.Tensor, top_k: int) -> None:
        super().__init__(ent2doc)
        self.top_k = top_k

        # 1. Tính toán trọng số IDF
        if ent2doc.is_sparse:
            frequency = torch.sparse.sum(ent2doc, dim=-1).to_dense()
            # TỐI ƯU CỐT LÕI 1: Chuyển vị ma trận ngay từ lúc khởi tạo
            # ent2doc có shape (Entities, Docs)
            # doc2ent có shape (Docs, Entities) -> Lưu sẵn dạng Sparse Coalesced
            self.doc2ent = ent2doc.t().coalesce()
        else:
            frequency = ent2doc.sum(dim=-1)
            self.doc2ent = ent2doc.t()

        self.idf_weight = 1 / frequency
        self.idf_weight[frequency == 0] = 0

    def __call__(self, ent_pred: torch.Tensor) -> torch.Tensor:
        """
        Rank documents based on top-k entity prediction with precise GPU profiling.
        """
        print("\n[PROFILER-RANKER] ================= START IDFWeightedTopKRanker =================")
        torch.cuda.synchronize()
        t_start = time.time()

        # ── Bước 1: Trích xuất Top K ──
        top_k_ent_pred = torch.topk(ent_pred, self.top_k, dim=-1)

        torch.cuda.synchronize()
        t_topk = time.time()
        print(f"[PROFILER-RANKER] Step 1 - Top-K Extraction took: {t_topk - t_start:.5f}s")

        # ── Bước 2: Tạo Dense Masked Tensor ──
        idf_weight = torch.gather(
            self.idf_weight.expand(ent_pred.shape[0], -1), 1, top_k_ent_pred.indices
        )
        masked_ent_pred = torch.zeros_like(ent_pred, device=ent_pred.device)
        masked_ent_pred.scatter_(1, top_k_ent_pred.indices, idf_weight)

        torch.cuda.synchronize()
        t_mask = time.time()
        print(f"[PROFILER-RANKER] Step 2 - Tensor Masking & Scatter took: {t_mask - t_topk:.5f}s")

        # ── Bước 3: Nhân Ma Trận (Tối ưu BLAS/cuSPARSE) ──
        # TOÁN HỌC: Doc_Pred = (Doc2Ent_Sparse @ Masked_Ent_Dense.T).T
        # Bằng cách này, ta ép PyTorch thực hiện Sparse @ Dense, tối đa hóa sức mạnh GPU.
        if self.doc2ent.is_sparse:
            doc_pred = torch.sparse.mm(self.doc2ent, masked_ent_pred.t()).t()
        else:
            doc_pred = torch.matmul(masked_ent_pred, self.ent2doc)

        torch.cuda.synchronize()
        t_mm = time.time()
        print(f"[PROFILER-RANKER] Step 3 - Matrix Multiply (Sparse @ Dense) took: {t_mm - t_mask:.5f}s")
        print(f"[PROFILER-RANKER] => TOTAL Doc Ranker executed in: {t_mm - t_start:.5f}s")
        print("[PROFILER-RANKER] ================= END IDFWeightedTopKRanker =================\n")

        return doc_pred
