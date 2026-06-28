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
from torch.utils import data as torch_data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.dataset import _repr, files_exist

from gfmrag.datasets.kg_dataset import KGDataset
from gfmrag.text_emb_models import BaseTextEmbModel
from gfmrag.utils import get_rank
from gfmrag.utils.qa_utils import entities_to_mask

logger = logging.getLogger(__name__)


# --- BƯỚC 1: CLASS SIÊU NHẸ THAY THẾ HUGGINGFACE DATASET ---
class SimpleDictDataset(torch_data.Dataset):
    """Wrapper siêu nhẹ thay thế HuggingFace Dataset để tránh lỗi OOM khi lưu file."""
    def __init__(self, data_dict):
        self.data_dict = data_dict
        # Lấy độ dài từ value đầu tiên trong dict
        self.length = len(next(iter(data_dict.values())))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data_dict.items()}
# -----------------------------------------------------------


class QADataset(InMemoryDataset):
    """Dataset class cho bài toán Hỏi đáp Y tế Tiếng Việt
    xây dựng trên nền Knowledge Graph BYT VN.

    Args:
        root (str): Thư mục gốc chứa dataset.
        data_name (str): Tên dataset (vd: vietnamese_medical).
        text_emb_model_cfgs (DictConfig): Config cho text embedding model.
        force_rebuild (bool): Buộc rebuild processed data. Defaults to False.
    """

    def __init__(
        self,
        root: str,
        data_name: str,
        text_emb_model_cfgs: DictConfig,
        force_rebuild: bool = False,
    ):
        self.name = data_name
        self.force_rebuild = force_rebuild
        self.text_emb_model_cfgs = text_emb_model_cfgs
        self.fingerprint = hashlib.md5(
            json.dumps(
                OmegaConf.to_container(text_emb_model_cfgs, resolve=True)
            ).encode()
        ).hexdigest()
        kg = KGDataset(root, data_name, text_emb_model_cfgs, force_rebuild)
        self.kg = kg[0]
        self.feat_dim = kg.feat_dim
        super().__init__(root, None, None)
        self.data = torch.load(self.processed_paths[0], weights_only=False)
        self.load_property()

    def __repr__(self) -> str:
        return f"{self.name}()"

    @property
    def raw_file_names(self) -> list:
        return ["train.json", "test.json"]

    @property
    def raw_dir(self) -> str:
        return os.path.join(str(self.root), str(self.name), "processed", "stage1")

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
        return "qa_data.pt"

    def load_property(self) -> None:
        """Load các property từ KG dataset với encoding UTF-8 cho tiếng Việt."""
        with open(
            os.path.join(self.processed_dir, "ent2id.json"),
            encoding="utf-8"
        ) as fin:
            self.ent2id = json.load(fin)

        with open(
            os.path.join(self.processed_dir, "rel2id.json"),
            encoding="utf-8"
        ) as fin:
            self.rel2id = json.load(fin)

        with open(
            os.path.join(str(self.root), str(self.name), "raw", "dataset_corpus.json"),
            encoding="utf-8"
        ) as fin:
            self.doc = json.load(fin)

        with open(
            os.path.join(self.raw_dir, "document2entities.json"),
            encoding="utf-8"
        ) as fin:
            self.doc2entities = json.load(fin)

        if os.path.exists(os.path.join(self.raw_dir, "train.json")):
            with open(
                os.path.join(self.raw_dir, "train.json"),
                encoding="utf-8"
            ) as fin:
                self.raw_train_data = json.load(fin)
        else:
            self.raw_train_data = []

        if os.path.exists(os.path.join(self.raw_dir, "test.json")):
            with open(
                os.path.join(self.raw_dir, "test.json"),
                encoding="utf-8"
            ) as fin:
                self.raw_test_data = json.load(fin)
        else:
            self.raw_test_data = []

        self.ent2docs = torch.load(
            os.path.join(self.processed_dir, "ent2doc.pt"),
            weights_only=True
        )
        self.id2doc = {i: doc for i, doc in enumerate(self.doc2entities)}

        logger.info(
            f"Loaded QA dataset '{self.name}':\n"
            f"  Entities: {len(self.ent2id)}\n"
            f"  Relations: {len(self.rel2id)}\n"
            f"  Documents: {len(self.doc)}\n"
            f"  Train samples: {len(self.raw_train_data)}\n"
            f"  Test samples: {len(self.raw_test_data)}"
        )

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
                f"Processing QA dataset {self.name} at rank {get_rank()}"
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

    def _build_ent2doc_sparse(
        self,
        doc2id: dict,
        num_nodes: int,
        n_docs: int,
    ) -> torch.Tensor:
        """
        Build sparse ent2doc tensor mà không tạo dense matrix trung gian.
        Tránh OOM với corpus lớn (130k nodes × 33k docs = 17GB nếu dùng dense).

        Returns:
            sparse_coo_tensor shape (num_nodes, n_docs)
        """
        entity_indices = []  # row indices (entity IDs)
        doc_indices = []     # col indices (doc IDs)

        for doc, entities in self.doc2entities.items():
            if doc not in doc2id:
                continue
            doc_idx = doc2id[doc]
            for ent in entities:
                if ent in self.ent2id:
                    entity_indices.append(self.ent2id[ent])
                    doc_indices.append(doc_idx)

        if len(entity_indices) == 0:
            logger.warning("Không có entity-document mapping nào được tìm thấy!")
            return torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros(0, dtype=torch.float),
                size=(num_nodes, n_docs),
            )

        indices = torch.tensor(
            [entity_indices, doc_indices],
            dtype=torch.long,
        )
        values = torch.ones(len(entity_indices), dtype=torch.float)

        ent2doc = torch.sparse_coo_tensor(
            indices,
            values,
            size=(num_nodes, n_docs),
        ).coalesce()

        logger.info(
            f"ent2doc sparse tensor: shape={ent2doc.shape}, "
            f"nnz={ent2doc._nnz()}, "
            f"density={ent2doc._nnz() / (num_nodes * n_docs) * 100:.4f}%"
        )
        return ent2doc

    def process(self) -> None:
        """Xử lý dataset QA y tế tiếng Việt.

        Tạo các file:
        - ent2doc.pt: Sparse tensor mapping entity → documents (không OOM)
        - qa_data.pt: Dataset đã xử lý với question embeddings và masks
        - text_emb_model_cfgs.json: Config embedding model
        """
        with open(
            os.path.join(self.processed_dir, "ent2id.json"),
            encoding="utf-8"
        ) as fin:
            self.ent2id = json.load(fin)

        with open(
            os.path.join(self.processed_dir, "rel2id.json"),
            encoding="utf-8"
        ) as fin:
            self.rel2id = json.load(fin)

        with open(
            os.path.join(self.raw_dir, "document2entities.json"),
            encoding="utf-8"
        ) as fin:
            self.doc2entities = json.load(fin)

        num_nodes = self.kg.num_nodes
        doc2id = {doc: i for i, doc in enumerate(self.doc2entities)}
        n_docs = len(self.doc2entities)

        logger.info(
            f"Building ent2doc sparse tensor: "
            f"{num_nodes} nodes × {n_docs} docs"
        )

        ent2doc = self._build_ent2doc_sparse(doc2id, num_nodes, n_docs)
        torch.save(ent2doc, os.path.join(self.processed_dir, "ent2doc.pt"))

        sample_id = []
        questions = []
        question_entities_masks = []
        supporting_entities_masks = []
        supporting_docs_masks = []
        num_samples = []

        for path in self.raw_paths:
            if not os.path.exists(path):
                num_samples.append(0)
                continue

            num_sample = 0
            skipped = 0

            with open(path, encoding="utf-8") as fin:
                data = json.load(fin)
                for index, item in enumerate(data):
                    question_entities = [
                        self.ent2id[x]
                        for x in item.get("question_entities", [])
                        if x in self.ent2id
                    ]
                    supporting_entities = [
                        self.ent2id[x]
                        for x in item.get("supporting_entities", [])
                        if x in self.ent2id
                    ]
                    supporting_docs = [
                        doc2id[doc]
                        for doc in item.get("supporting_facts", [])
                        if doc in doc2id
                    ]

                    # Nếu supporting_entities rỗng → dùng question_entities
                    if len(supporting_entities) == 0:
                        supporting_entities = question_entities

                    # Chỉ skip nếu question_entities hoặc supporting_docs rỗng
                    if len(question_entities) == 0 or len(supporting_docs) == 0:
                        skipped += 1
                        logger.debug(
                            f"Bỏ qua sample {index}: "
                            f"question_entities={len(question_entities)}, "
                            f"supporting_docs={len(supporting_docs)}"
                        )
                        continue

                    num_sample += 1
                    sample_id.append(index)
                    questions.append(item["question"])
                    question_entities_masks.append(
                        entities_to_mask(question_entities, num_nodes)
                    )
                    supporting_entities_masks.append(
                        entities_to_mask(supporting_entities, num_nodes)
                    )
                    supporting_docs_masks.append(
                        entities_to_mask(supporting_docs, n_docs)
                    )

            num_samples.append(num_sample)
            logger.info(
                f"{osp.basename(path)}: "
                f"{num_sample} samples hợp lệ, "
                f"{skipped} samples bị bỏ qua"
            )

        logger.info(
            f"Generating embeddings cho {len(questions)} câu hỏi tiếng Việt..."
        )

        # Handle trường hợp không có sample nào
        if len(questions) == 0:
            logger.warning(
                "Không có sample hợp lệ nào! Kiểm tra lại format test.json:\n"
                "  - question_entities phải khớp với entity trong KG\n"
                "  - supporting_facts phải khớp với title trong document2entities"
            )
            # Khởi tạo dataset trống với class SimpleDictDataset mới
            empty_data_dict = {
                "question_embeddings": torch.empty((0, 1024)),
                "question_entities_masks": torch.empty((0, num_nodes)),
                "supporting_entities_masks": torch.empty((0, num_nodes)),
                "supporting_docs_masks": torch.empty((0, n_docs)),
                "sample_id": torch.empty((0,), dtype=torch.long),
            }
            empty_dataset = SimpleDictDataset(empty_data_dict)
            splits = [
                torch_data.Subset(empty_dataset, [])
                for _ in num_samples
            ]
            torch.save(splits, self.processed_paths[0])
            return

        text_emb_model: BaseTextEmbModel = instantiate(self.text_emb_model_cfgs)
        question_embeddings = text_emb_model.encode(
            questions, is_query=True
        ).cpu()
        logger.info(f"Question embeddings shape: {question_embeddings.shape}")

        question_entities_masks = torch.stack(question_entities_masks)
        supporting_entities_masks = torch.stack(supporting_entities_masks)
        supporting_docs_masks = torch.stack(supporting_docs_masks)
        sample_id = torch.tensor(sample_id, dtype=torch.long)

        # --- BƯỚC 2: TỐI ƯU MEMORY BẰNG THUẦN PYTORCH TENSORS ---
        logger.info("Initializing SimpleDictDataset directly from Tensors to avoid Pickle OOM...")
        data_dict = {
            "question_embeddings": question_embeddings,
            "question_entities_masks": question_entities_masks,
            "supporting_entities_masks": supporting_entities_masks,
            "supporting_docs_masks": supporting_docs_masks,
            "sample_id": sample_id,
        }

        # Sử dụng class siêu nhẹ vừa tạo ở đầu file thay cho HuggingFace Dataset
        dataset = SimpleDictDataset(data_dict)
        # --------------------------------------------------------

        offset = 0
        splits = []
        for num_sample in num_samples:
            split = torch_data.Subset(dataset, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample

        torch.save(splits, self.processed_paths[0])

        with open(
            self.processed_dir + "/text_emb_model_cfgs.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                OmegaConf.to_container(self.text_emb_model_cfgs),
                f,
                ensure_ascii=False,
                indent=4,
            )

        logger.info(
            f"QA dataset processed xong:\n"
            f"  Total samples: {sum(num_samples)}\n"
            f"  Splits: {num_samples}\n"
            f"  Question embedding dim: {question_embeddings.shape[1]}\n"
            f"  Saved to: {self.processed_paths[0]}"
        )