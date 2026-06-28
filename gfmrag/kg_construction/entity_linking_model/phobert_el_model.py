import os
import torch
import hashlib
import shutil
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import util
import torch.nn.functional as F
from .base_model import BaseELModel
from torch.utils.data import DataLoader


class PhoBertELModel(BaseELModel):
    def __init__(
            self,
            model_name="vinai/phobert-base",
            root="tmp",
            use_cache=True,
            normalize=True,
            batch_size=32,
            force=False,
            device=None
    ):
        self.model_name = model_name
        self.root = os.path.join(root, f"{self.model_name.replace('/', '_')}_phobert_cache")
        self.use_cache = use_cache
        self.normalize = normalize
        self.batch_size = batch_size
        self.force = force
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        if self.use_cache and not os.path.exists(self.root):
            os.makedirs(self.root)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def encode(self, texts, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        all_embeddings = []
        self.model.eval()

        dataloader = DataLoader(texts, batch_size=batch_size)
        for batch in dataloader:
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]
                all_embeddings.append(embeddings.cpu())

        embeddings = torch.cat(all_embeddings, dim=0)

        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def index(self, entity_list):
        self.entity_list = entity_list
        fingerprint = hashlib.md5("".join(entity_list).encode()).hexdigest()
        cache_file = f"{self.root}/{fingerprint}.pt"

        if os.path.exists(cache_file) and not self.force:
            self.entity_embeddings = torch.load(
                cache_file, map_location=self.device
            ).to(self.device)
            return

        self.entity_embeddings = self.encode(entity_list).to(self.device)

        if self.use_cache:
            torch.save(self.entity_embeddings.cpu(), cache_file)

    def __call__(self, ner_entity_list, topk=1):
        query_embeddings = self.encode(ner_entity_list).to(self.device)
        linked_entity_dict = {}

        scores = util.pytorch_cos_sim(query_embeddings, self.entity_embeddings)
        top_k_scores, top_k_values = torch.topk(scores, topk, dim=-1)

        for i in range(len(ner_entity_list)):
            linked_entity_dict[ner_entity_list[i]] = []

            sorted_score = top_k_scores[i]
            sorted_indices = top_k_values[i]
            max_score = sorted_score[0].item()

            for score, top_k_index in zip(sorted_score, sorted_indices):
                linked_entity_dict[ner_entity_list[i]].append(
                    {
                        "entity": self.entity_list[top_k_index],
                        "score": score.item(),
                        "norm_score": score.item() / max_score,
                    }
                )

        return linked_entity_dict