import json
from gfmrag.evaluation.base_evaluator import BaseEvaluator

class RetrievalEvaluator(BaseEvaluator):
    def __init__(self, prediction_file: str, precomputed_chunks_file: str = None):
        super().__init__(prediction_file)
        # Bỏ hoàn toàn logic load file precomputed vì chỉ cần đối chiếu trực tiếp chunk_id

    def evaluate(self, top_k_list: list = None) -> dict:
        if top_k_list is None:
            top_k_list = [1, 2, 3, 4, 5, 10]

        # Khởi tạo metrics chỉ chứa chunk_recall
        metrics = {f"chunk_recall@{k}": 0.0 for k in top_k_list}
        valid_samples = 0

        for sample in self.data:
            supp_facts = sample.get("supporting_facts", [])
            if not supp_facts:
                continue

            valid_samples += 1

            # 1. Tập Gold: Lấy trực tiếp danh sách chunk_id từ supporting_facts
            gold_chunks = set(str(sf) for sf in supp_facts)

            # 2. Xử lý các chunks mà hệ thống (RAG) lấy về
            retrieved = sample.get("retrieved_docs", [])

            for k in top_k_list:
                pred_chunks = set()

                # Lấy ra các chunk_id nằm trong top k
                for r in retrieved[:k]:
                    c_id = str(r.get("chunk_id", ""))
                    if c_id:
                        pred_chunks.add(c_id)

                # Tính Chunk-level Recall@K
                if len(gold_chunks) > 0:
                    chunk_hits = len(gold_chunks.intersection(pred_chunks))
                    metrics[f"chunk_recall@{k}"] += chunk_hits / len(gold_chunks)

        # Tính trung bình (Average) cho toàn bộ tập dữ liệu
        if valid_samples > 0:
            for k in top_k_list:
                metrics[f"chunk_recall@{k}"] /= valid_samples

        return metrics