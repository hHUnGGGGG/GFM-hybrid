import io
import json
import logging
import os
import sys

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from gfmrag_hybrid import GFMRetriever
from gfmrag_hybrid.evaluation import RetrievalEvaluator
from gfmrag_hybrid.llms import BaseLanguageModel
from gfmrag_hybrid.prompt_builder import QAPromptBuilder
from gfmrag_hybrid.ultra import query_utils

# Fix Windows console encoding cho tiếng Việt
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

logger = logging.getLogger(__name__)

# Từ kết thúc agent loop — tiếng Việt
STOP_PHRASE = "Vậy câu trả lời là:"


def agent_reasoning(
    cfg: DictConfig,
    gfmrag_retriever: GFMRetriever,
    llm: BaseLanguageModel,
    qa_prompt_builder: QAPromptBuilder,
    query: str,
) -> dict:
    """
    Agent reasoning loop cho câu hỏi y tế tiếng Việt.
    Thay thế "So the answer is:" bằng "Vậy câu trả lời là:"
    """
    step = 1
    current_query = query
    thoughts: list[str] = []
    retrieved_docs = gfmrag_retriever.retrieve(current_query, top_k=cfg.test.top_k)
    logs = []

    while step <= cfg.test.max_steps:
        message = qa_prompt_builder.build_input_prompt(
            current_query, retrieved_docs, thoughts
        )
        response = llm.generate_sentence(message)

        if isinstance(response, Exception):
            raise response from None

        thoughts.append(response)
        logs.append(
            {
                "step": step,
                "query": current_query,
                "retrieved_docs": retrieved_docs,
                "response": response,
                "thoughts": thoughts,
            }
        )

        # FIX 1: kiểm tra stop phrase tiếng Việt thay vì "So the answer is:"
        if STOP_PHRASE in response:
            break

        step += 1

        new_ret_docs = gfmrag_retriever.retrieve(response, top_k=cfg.test.top_k)

        retrieved_docs_dict = {doc["title"]: doc for doc in retrieved_docs}
        for doc in new_ret_docs:
            if doc["title"] in retrieved_docs_dict:
                if doc["norm_score"] > retrieved_docs_dict[doc["title"]]["norm_score"]:
                    retrieved_docs_dict[doc["title"]]["score"] = doc["score"]
                    retrieved_docs_dict[doc["title"]]["norm_score"] = doc["norm_score"]
            else:
                retrieved_docs_dict[doc["title"]] = doc

        retrieved_docs = sorted(
            retrieved_docs_dict.values(),
            key=lambda x: x["norm_score"],
            reverse=True,
        )
        retrieved_docs = retrieved_docs[: cfg.test.top_k]

    final_response = " ".join(thoughts)
    return {
        "response": final_response,
        "retrieved_docs": retrieved_docs,
        "logs": logs,
    }


@hydra.main(
    config_path="config",
    config_name="stage3_qa_ircot_inference_vietnamese_medical",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    output_dir = HydraConfig.get().runtime.output_dir

    # FIX 2: log config an toàn — tránh UnicodeEncodeError trên Windows
    try:
        logger.info(f"Config:\n {OmegaConf.to_yaml(cfg)}")
    except UnicodeEncodeError:
        logger.info("Config loaded (unicode log skipped on Windows)")

    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Output directory: {output_dir}")

    gfmrag_retriever = GFMRetriever.from_config(cfg)
    llm = instantiate(cfg.llm)
    agent_prompt_builder = QAPromptBuilder(cfg.agent_prompt)
    qa_prompt_builder = QAPromptBuilder(cfg.qa_prompt)

    test_data = gfmrag_retriever.qa_data.raw_test_data
    max_samples = (
        cfg.test.max_test_samples
        if cfg.test.max_test_samples > 0
        else len(test_data)
    )

    logger.info(f"Tổng số test samples: {len(test_data)}, chạy: {max_samples}")

    # Load previous results nếu resume
    processed_data = {}
    if cfg.test.resume:
        logger.info(f"Resuming from {cfg.test.resume}")
        try:
            # FIX 3: encoding="utf-8" khi đọc file resume
            with open(cfg.test.resume, encoding="utf-8") as f:
                for line in f:
                    result = json.loads(line)
                    processed_data[result["id"]] = result
            logger.info(f"Loaded {len(processed_data)} processed samples")
        except Exception as e:
            logger.error(f"Could not resume: {e}")

    # FIX 3: encoding="utf-8" khi ghi prediction.jsonl
    prediction_path = os.path.join(output_dir, "prediction.jsonl")
    with open(prediction_path, "w", encoding="utf-8") as f:
        for i in tqdm(range(max_samples), desc="Inference"):
            if i >= len(test_data):
                break

            sample = test_data[i]
            query = sample["question"]

            if sample["id"] in processed_data:
                result = processed_data[sample["id"]]
            else:
                try:
                    result = agent_reasoning(
                        cfg,
                        gfmrag_retriever,
                        llm,
                        agent_prompt_builder,
                        query,
                    )

                    # Generate final QA response
                    retrieved_docs = result["retrieved_docs"]
                    message = qa_prompt_builder.build_input_prompt(
                        query, retrieved_docs
                    )
                    qa_response = llm.generate_sentence(message)

                    result = {
                        "id": sample["id"],
                        "question": sample["question"],
                        "answer": sample["answer"],
                        "answer_aliases": sample.get("answer_aliases", []),
                        "supporting_facts": sample["supporting_facts"],
                        "response": qa_response,
                        "retrieved_docs": retrieved_docs,
                        "logs": result["logs"],
                    }

                except Exception as e:
                    # FIX 4: không crash toàn bộ nếu 1 sample lỗi
                    logger.error(f"Lỗi ở sample {i} (id={sample.get('id')}): {e}")
                    result = {
                        "id": sample.get("id", f"error_{i}"),
                        "question": query,
                        "answer": sample.get("answer", ""),
                        "answer_aliases": sample.get("answer_aliases", []),
                        "supporting_facts": sample.get("supporting_facts", []),
                        "response": "ERROR",
                        "retrieved_docs": [],
                        "logs": [],
                        "error": str(e),
                    }

            # FIX 3: ensure_ascii=False để giữ tiếng Việt trong output
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

    logger.info(f"Predictions saved to {prediction_path}")

    # Evaluation
    try:
        evaluator = instantiate(cfg.qa_evaluator, prediction_file=prediction_path)
        metrics = evaluator.evaluate()
        query_utils.print_metrics(metrics, logger)
    except Exception as e:
        logger.error(f"QA evaluation lỗi: {e}")

    try:
        retrieval_evaluator = RetrievalEvaluator(prediction_file=prediction_path)
        retrieval_metrics = retrieval_evaluator.evaluate()
        query_utils.print_metrics(retrieval_metrics, logger)
    except Exception as e:
        logger.error(f"Retrieval evaluation lỗi: {e}")


if __name__ == "__main__":
    main()