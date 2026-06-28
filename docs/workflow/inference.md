GFM-RAG can be directly used for retrieval on a given dataset without fine-tuning. We provide an easy-to-use [GFMRetriever][gfmrag.GFMRetriever] interface for inference.

!!! NOTE
    We have already released the [pre-trained model](https://huggingface.co/rmanluo/GFM-RAG-8M), which can be used directly for retrieval. The model will be automatically downloaded by specifying it in the configuration.
    ```yaml
    graph_retriever:
      model_path: rmanluo/GFM-RAG-8M
    ```

## Config

You need to create a configuration file for inference.

??? example "gfmrag/workflow/config/stage3_qa_ircot_inference.yaml"

    ```yaml title="gfmrag/workflow/config/stage3_qa_ircot_inference.yaml"
    --8<-- "gfmrag/workflow/config/stage3_qa_ircot_inference.yaml"
    ```

Details of the configuration parameters are explained in the [GFM-RAG Configuration][gfm-rag-configuration] page.

## Initialize GFMRetriever

You can initialize the GFMRetriever with the following code. It will load the pre-trained GFM-RAG model and the KG-index for retrieval.

```python
import logging
import os

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from gfmrag import GFMRetriever

logger = logging.getLogger(__name__)


@hydra.main(
    config_path="config", config_name="stage3_qa_ircot_inference", version_base=None
)
def main(cfg: DictConfig) -> None:
    output_dir = HydraConfig.get().runtime.output_dir
    logger.info(f"Config:\n {OmegaConf.to_yaml(cfg)}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Output directory: {output_dir}")

    gfmrag_retriever = GFMRetriever.from_config(cfg)
```

## Document Retrieval

You can use GFM-RAG retriever to reason over the KG-index and obtain documents for a given query.
```python
docs = retriever.retrieve("Who is the president of France?", top_k=5)
```

## Question Answering

```python
from hydra.utils import instantiate
from gfmrag.llms import BaseLanguageModel
from gfmrag.prompt_builder import QAPromptBuilder

llm = instantiate(cfg.llm)
qa_prompt_builder = QAPromptBuilder(cfg.qa_prompt)

message = qa_prompt_builder.build_input_prompt(current_query, retrieved_docs)
answer = llm.generate_sentence(message)  # Answer: "Emmanuel Macron"
```

## GFM-RAG + Agent for Multi-step Retrieval
You can also integrate the GFM-RAG with arbitrary reasoning agents to perform multi-step RAG. Here is an example of [IRCOT](https://arxiv.org/abs/2212.10509) + GFM-RAG:

You can run the following command to perform multi-step reasoning:

??? example "gfmrag/workflow/stage3_qa_ircot_inference.py"

    <!-- blacken-docs:off -->
    ```python title="gfmrag/workflow/stage3_qa_ircot_inference.py"
    --8<-- "gfmrag/workflow/stage3_qa_ircot_inference.py"
    ```
    <!-- blacken-docs:on -->

```bash
python -m gfmrag.workflow.stage3_qa_ircot_inference
```

You can overwrite the configuration like this:

```bash
python -m gfmrag.workflow.stage3_qa_ircot_inference test.max_steps=3
```

## Batch Retrieval
You can also perform batch retrieval with GFM-RAG with multi GPUs supports by running the following command:

??? example "gfmrag/workflow/config/stage3_qa_inference.yaml"

    ```yaml title="gfmrag/workflow/config/stage3_qa_inference.yaml"
    --8<-- "gfmrag/workflow/config/stage3_qa_inference.yaml"
    ```

??? example "gfmrag/workflow/stage3_qa_inference.py"

    <!-- blacken-docs:off -->
    ```python title="gfmrag/workflow/stage3_qa_inference.py"
    --8<-- "gfmrag/workflow/stage3_qa_inference.py"
    ```
    <!-- blacken-docs:on -->

```bash
python -m gfmrag.workflow.stage3_qa_inference
# Multi-GPU retrieval
torchrun --nproc_per_node=4 -m gfmrag.workflow.stage3_qa_inference
```

You can overwrite the configuration like this:

```bash
python -m gfmrag.workflow.stage3_qa_inference test.retrieval_batch_size=4
```
