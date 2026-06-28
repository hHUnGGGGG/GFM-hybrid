import hashlib
import json
import logging
import os
import os.path as osp
import sys
import warnings

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.dataset import _repr, files_exist

from gfmrag_hybrid.kg_construction.utils import KG_DELIMITER
from gfmrag_hybrid.text_emb_models import BaseTextEmbModel
from gfmrag_hybrid.utils import get_rank

logger = logging.getLogger(__name__)


class KGDataset(InMemoryDataset):
    """Dataset class cho Knowledge Graph y tế tiếng Việt.

    Xử lý dữ liệu KG từ corpus BYT VN, bao gồm các triple
    (entity, relation, entity) với encoding UTF-8 cho tiếng Việt.

    Args:
        root (str): Thư mục gốc chứa dataset.
        data_name (str): Tên dataset (vd: vietnamese_medical).
        text_emb_model_cfgs (DictConfig): Config cho text embedding model.
        force_rebuild (bool): Buộc rebuild processed data. Defaults to False.
    """

    delimiter = KG_DELIMITER

    def __init__(
        self,
        root: str,
        data_name: str,
        text_emb_model_cfgs: DictConfig,
        force_rebuild: bool = False,
        **kwargs: str,
    ) -> None:
        self.name = data_name
        self.force_rebuild = force_rebuild
        self.fingerprint = hashlib.md5(
            json.dumps(
                OmegaConf.to_container(text_emb_model_cfgs, resolve=True)
            ).encode()
        ).hexdigest()
        self.text_emb_model_cfgs = text_emb_model_cfgs
        super().__init__(root, None, None)
        self.data, self.slices = torch.load(
            self.processed_paths[0], weights_only=False
        )
        self.feat_dim = self._data.rel_emb.size(1)

    @property
    def raw_file_names(self) -> list:
        return ["kg.txt"]

    def load_file(
        self, triplet_file: str, inv_entity_vocab: dict, inv_rel_vocab: dict
    ) -> dict:
        """Load file KG tiếng Việt với encoding UTF-8.

        Args:
            triplet_file (str): Đường dẫn đến kg.txt
            inv_entity_vocab (dict): Mapping entity → ID
            inv_rel_vocab (dict): Mapping relation → ID

        Returns:
            dict: triplets, num_node, num_relation, inv_entity_vocab, inv_rel_vocab
        """
        triplets = []
        entity_cnt, rel_cnt = len(inv_entity_vocab), len(inv_rel_vocab)

        # FIX 1: encoding="utf-8" để đọc đúng tiếng Việt trong kg.txt
        with open(triplet_file, encoding="utf-8") as fin:
            for line_num, line in enumerate(fin, 1):
                try:
                    u, r, v = (
                        line.split()
                        if self.delimiter is None
                        else line.strip().split(self.delimiter)
                    )
                except Exception as e:
                    # FIX 2: log rõ line number để debug dễ hơn với corpus tiếng Việt
                    logger.warning(
                        f"Bỏ qua dòng {line_num} bị lỗi format: "
                        f"'{line.strip()[:50]}...' — {e}"
                    )
                    continue

                # FIX 3: strip() để loại bỏ khoảng trắng thừa
                # thường gặp sau khi OCR PDF tiếng Việt
                u = u.strip()
                r = r.strip()
                v = v.strip()

                # Bỏ qua triple rỗng sau khi strip
                if not u or not r or not v:
                    logger.warning(f"Bỏ qua triple rỗng ở dòng {line_num}")
                    continue

                if u not in inv_entity_vocab:
                    inv_entity_vocab[u] = entity_cnt
                    entity_cnt += 1
                if v not in inv_entity_vocab:
                    inv_entity_vocab[v] = entity_cnt
                    entity_cnt += 1
                if r not in inv_rel_vocab:
                    inv_rel_vocab[r] = rel_cnt
                    rel_cnt += 1

                u, r, v = inv_entity_vocab[u], inv_rel_vocab[r], inv_entity_vocab[v]
                triplets.append((u, v, r))

        logger.info(
            f"Loaded {len(triplets)} triples, "
            f"{len(inv_entity_vocab)} entities, "
            f"{len(inv_rel_vocab)} relations"
        )

        return {
            "triplets": triplets,
            "num_node": len(inv_entity_vocab),
            "num_relation": rel_cnt,
            "inv_entity_vocab": inv_entity_vocab,
            "inv_rel_vocab": inv_rel_vocab,
        }

    def _process(self) -> None:
        f = osp.join(self.processed_dir, "pre_transform.pt")
        if osp.exists(f) and torch.load(f, weights_only=False) != _repr(
            self.pre_transform
        ):
            warnings.warn(
                f"The `pre_transform` argument differs from the one used in "
                f"the pre-processed version of this dataset. If you want to "
                f"make use of another pre-processing technique, make sure to "
                f"delete '{self.processed_dir}' first",
                stacklevel=1,
            )

        f = osp.join(self.processed_dir, "pre_filter.pt")
        if osp.exists(f) and torch.load(f, weights_only=False) != _repr(
            self.pre_filter
        ):
            warnings.warn(
                f"The `pre_filter` argument differs from the one used in "
                f"the pre-processed version of this dataset. If you want to "
                f"make use of another pre-filtering technique, make sure to "
                f"delete '{self.processed_dir}' first",
                stacklevel=1,
            )

        if self.force_rebuild or not files_exist(self.processed_paths):
            logger.warning(
                f"Processing KG dataset {self.name} at rank {get_rank()}"
            )
            if self.log and "pytest" not in sys.modules:
                print("Processing...", file=sys.stderr)

            os.makedirs(self.processed_dir, exist_ok=True)
            self.process()

            path = osp.join(self.processed_dir, "pre_transform.pt")
            torch.save(_repr(self.pre_transform), path)
            path = osp.join(self.processed_dir, "pre_filter.pt")
            torch.save(_repr(self.pre_filter), path)

            if self.log and "pytest" not in sys.modules:
                print("Done!", file=sys.stderr)

    def process(self) -> None:
        """Xử lý KG dataset y tế tiếng Việt.

        Tạo các file:
        - ent2id.json: Entity → ID (tiếng Việt, UTF-8)
        - rel2id.json: Relation → ID (tiếng Việt, UTF-8)
        - text_emb_model_cfgs.json: Config embedding model
        - data.pt: PyG graph data đã xử lý
        """
        kg_file = self.raw_paths[0]
        kg_result = self.load_file(kg_file, inv_entity_vocab={}, inv_rel_vocab={})

        num_node = kg_result["num_node"]
        num_relations = kg_result["num_relation"]
        kg_triplets = kg_result["triplets"]

        train_target_edges = torch.tensor(
            [[t[0], t[1]] for t in kg_triplets], dtype=torch.long
        ).t()
        train_target_etypes = torch.tensor([t[2] for t in kg_triplets])

        # Thêm inverse edges
        train_edges = torch.cat(
            [train_target_edges, train_target_edges.flip(0)], dim=1
        )
        train_etypes = torch.cat(
            [train_target_etypes, train_target_etypes + num_relations]
        )

        # FIX 4: ensure_ascii=False để giữ tiếng Việt trong ent2id.json
        with open(
            self.processed_dir + "/ent2id.json", "w", encoding="utf-8"
        ) as f:
            json.dump(
                kg_result["inv_entity_vocab"], f,
                ensure_ascii=False,   # giữ ký tự tiếng Việt
                indent=2,
            )

        rel2id = kg_result["inv_rel_vocab"]
        id2rel = {v: k for k, v in rel2id.items()}
        for etype in train_etypes:
            if etype.item() >= num_relations:
                raw_etype = etype - num_relations
                raw_rel = id2rel[raw_etype.item()]
                rel2id["inverse_" + raw_rel] = etype.item()

        # FIX 4: ensure_ascii=False cho rel2id.json
        with open(
            self.processed_dir + "/rel2id.json", "w", encoding="utf-8"
        ) as f:
            json.dump(
                rel2id, f,
                ensure_ascii=False,   # giữ relation tiếng Việt như "có thể gây", "điều trị"
                indent=2,
            )

        # Generate relation embeddings
        logger.info(
            f"Generating relation embeddings cho {len(rel2id)} relations..."
        )
        text_emb_model: BaseTextEmbModel = instantiate(self.text_emb_model_cfgs)
        rel_emb = text_emb_model.encode(
            list(rel2id.keys()), is_query=False
        ).cpu()
        logger.info(f"Relation embeddings shape: {rel_emb.shape}")

        kg_data = Data(
            edge_index=train_edges,
            edge_type=train_etypes,
            num_nodes=num_node,
            target_edge_index=train_target_edges,
            target_edge_type=train_target_etypes,
            num_relations=num_relations * 2,
            rel_emb=rel_emb,
        )

        torch.save((self.collate([kg_data])), self.processed_paths[0])

        # FIX 4: ensure_ascii=False cho config file
        with open(
            self.processed_dir + "/text_emb_model_cfgs.json", "w", encoding="utf-8"
        ) as f:
            json.dump(
                OmegaConf.to_container(self.text_emb_model_cfgs),
                f,
                ensure_ascii=False,
                indent=4,
            )

        logger.info(
            f"KG dataset processed xong:\n"
            f"  Nodes: {num_node}\n"
            f"  Relations: {num_relations} (x2 với inverse = {num_relations*2})\n"
            f"  Edges: {len(kg_triplets)}\n"
            f"  Saved to: {self.processed_paths[0]}"
        )

    def __repr__(self) -> str:
        return f"{self.name}()"

    @property
    def num_relations(self) -> int:
        return int(self.data.edge_type.max()) + 1

    @property
    def raw_dir(self) -> str:
        return os.path.join(
            str(self.root), str(self.name), "processed", "stage1"
        )

    @property
    def processed_dir(self) -> str:
        return os.path.join(
            str(self.root),
            str(self.name),
            "processed",
            "stage2",
            self.fingerprint,
        )

    @property
    def processed_file_names(self) -> str:
        return "data.pt"