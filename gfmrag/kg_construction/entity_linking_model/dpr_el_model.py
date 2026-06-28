import hashlib
import os
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

from .base_model import BaseELModel


class DPRELModel(BaseELModel):
    """
    Entity Linking Model based on Dense Passage Retrieval (DPR).

    Args:
        model_name (str): Name or path of the SentenceTransformer model to use
        root (str, optional): Root directory for caching embeddings. Defaults to "tmp".
        use_cache (bool, optional): Whether to cache and reuse entity embeddings. Defaults to True.
        normalize (bool, optional): Whether to L2-normalize embeddings. Defaults to True.
        batch_size (int, optional): Batch size for encoding. Defaults to 32.
        query_instruct (str, optional): Instruction/prompt prefix for query encoding. Defaults to "".
        passage_instruct (str, optional): Instruction/prompt prefix for passage encoding. Defaults to "".
        model_kwargs (dict, optional): Additional kwargs to pass to SentenceTransformer. Defaults to None.
        sim_batch_size (int, optional): Batch size for similarity matrix computation. Defaults to 512.
        chunk_size (int, optional): Chunk size for processing ner_entity_list. Defaults to 100.
    """

    def __init__(
        self,
        model_name: str,
        root: str = "tmp",
        use_cache: bool = True,
        normalize: bool = True,
        batch_size: int = 32,
        query_instruct: str = "",
        passage_instruct: str = "",
        model_kwargs: dict | None = None,
        sim_batch_size: int = 512,
        chunk_size: int = 50,
    ) -> None:
        self.model_name = model_name
        self.use_cache = use_cache
        self.normalize = normalize
        self.batch_size = batch_size
        self.sim_batch_size = sim_batch_size
        self.chunk_size = chunk_size
        self.root = os.path.join(root, f"{self.model_name.replace('/', '_')}_dpr_cache")
        if self.use_cache and not os.path.exists(self.root):
            os.makedirs(self.root)
        self.model = SentenceTransformer(
            model_name, trust_remote_code=True, model_kwargs=model_kwargs
        )
        self.query_instruct = query_instruct
        self.passage_instruct = passage_instruct

    def index(self, entity_list: list) -> None:
        """
        Index a list of entities by encoding them into embeddings and optionally caching the results.

        Args:
            entity_list (list): A list of strings representing entities to be indexed.
        """
        self.entity_list = entity_list
        fingerprint = hashlib.md5("".join(entity_list).encode()).hexdigest()
        cache_file = f"{self.root}/{fingerprint}.pt"

        encode_device = "cuda" if torch.cuda.is_available() else "cpu"

        if os.path.exists(cache_file):
            # Load về CPU để tránh OOM khi matrix quá lớn
            self.entity_embeddings = torch.load(
                cache_file,
                map_location="cpu",
                weights_only=True,
            )
        else:
            self.entity_embeddings = self.model.encode(
                entity_list,
                device=encode_device,
                convert_to_tensor=True,
                show_progress_bar=True,
                prompt=self.passage_instruct,
                normalize_embeddings=self.normalize,
                batch_size=self.batch_size,
            )
            # Chuyển về CPU trước khi cache để tránh OOM
            self.entity_embeddings = self.entity_embeddings.cpu()

            if self.use_cache:
                torch.save(self.entity_embeddings, cache_file)

    def _compute_similarity_batched(
        self,
        ner_entity_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Tính cosine similarity theo batch trên CPU để tránh OOM.

        Args:
            ner_entity_embeddings: Tensor shape [N, dim]

        Returns:
            scores: Tensor shape [N, M] với M là số entity trong KG
        """
        ner_emb_cpu = ner_entity_embeddings.cpu()
        entity_emb_cpu = self.entity_embeddings.cpu()

        all_scores = []
        for i in range(0, len(ner_emb_cpu), self.sim_batch_size):
            batch = ner_emb_cpu[i : i + self.sim_batch_size]
            batch_scores = batch @ entity_emb_cpu.T  # [batch, M]
            all_scores.append(batch_scores)

        return torch.cat(all_scores, dim=0)  # [N, M]

    def _process_chunk(self, chunk: list, topk: int) -> dict:
        """
        Encode và link một chunk nhỏ của ner_entity_list.

        Args:
            chunk (list): Subset của ner_entity_list
            topk (int): Số top matches cần trả về

        Returns:
            dict: Kết quả entity linking cho chunk này
        """
        # Force CPU để tránh OOM hoàn toàn
        ner_entity_embeddings = self.model.encode(
            chunk,
            device="cpu",
            convert_to_tensor=True,
            prompt=self.query_instruct,
            normalize_embeddings=self.normalize,
            batch_size=self.batch_size,
        )

        # Tính similarity theo batch trên CPU
        scores = self._compute_similarity_batched(ner_entity_embeddings)

        # Đảm bảo topk không vượt quá số entity trong KG
        actual_topk = min(topk, len(self.entity_list))
        top_k_scores, top_k_values = torch.topk(scores, actual_topk, dim=-1)

        chunk_result: dict[str, list] = {}
        for i in range(len(chunk)):
            chunk_result[chunk[i]] = []

            sorted_score = top_k_scores[i]
            sorted_indices = top_k_values[i]
            max_score = sorted_score[0].item()

            # Tránh chia cho 0 nếu max_score = 0
            max_score = max_score if max_score != 0 else 1e-9

            for score, top_k_index in zip(sorted_score, sorted_indices):
                chunk_result[chunk[i]].append(
                    {
                        "entity": self.entity_list[top_k_index],
                        "score": score.item(),
                        "norm_score": score.item() / max_score,
                    }
                )
        return chunk_result

    def __call__(self, ner_entity_list: list, topk: int = 1) -> dict:
        """
        Performs entity linking by matching input entities with pre-encoded entity embeddings.
        Xử lý theo chunk để tránh OOM với corpus lớn.

        Args:
            ner_entity_list (list): List of named entities to link
            topk (int, optional): Number of top matches to return for each entity. Defaults to 1.

        Returns:
            dict: Dictionary mapping each input entity to its linked candidates.
        """
        results: dict[str, list] = {}

        for i in range(0, len(ner_entity_list), self.chunk_size):
            chunk = ner_entity_list[i : i + self.chunk_size]
            chunk_results = self._process_chunk(chunk, topk)
            results.update(chunk_results)

        return results


class NVEmbedV2ELModel(DPRELModel):
    """
    A DPR-based Entity Linking model specialized for NVEmbed V2 embeddings.
    """

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            **kwargs,
        )
        self.model.max_seq_length = 32768
        self.model.tokenizer.padding_side = "right"

    def add_eos(self, input_examples: list[str]) -> list[str]:
        """Appends EOS token to each input example."""
        input_examples = [
            input_example + self.model.tokenizer.eos_token
            for input_example in input_examples
        ]
        return input_examples

    def __call__(self, ner_entity_list: list, *args: Any, **kwargs: Any) -> dict:
        """Execute entity linking with EOS tokens."""
        ner_entity_list = self.add_eos(ner_entity_list)
        return super().__call__(ner_entity_list, *args, **kwargs)