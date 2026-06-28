import io
import json
import logging
import os
import sys
import string
from rank_bm25 import BM25Okapi
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from gfmrag import GFMRetriever
from gfmrag.evaluation import RetrievalEvaluator
from gfmrag.llms import BaseLanguageModel
from gfmrag.prompt_builder import QAPromptBuilder
from gfmrag.ultra import query_utils

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

logger = logging.getLogger(__name__)

# Stop phrase for MuSiQue dataset (English)
STOP_PHRASE = "So the answer is:"


def agent_reasoning(
    cfg: DictConfig,
    gfmrag_retriever: GFMRetriever,
    llm: BaseLanguageModel,
    qa_prompt_builder: QAPromptBuilder,
    query: str,
) -> dict:
    """
    Agent reasoning loop for MuSiQue dataset.
    Stops when the LLM generates the STOP_PHRASE.
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

        # Check for English stop phrase
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
    config_name="stage3_qa_ircot_inference", # Äáº£m báº£o file yaml cá»§a báº¡n cÃ³ tÃªn khá»›p vá»›i chuá»—i nÃ y
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    output_dir = HydraConfig.get().runtime.output_dir

    # Log config safely - prevents UnicodeEncodeError on Windows
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

    logger.info(f"Total test samples: {len(test_data)}, running: {max_samples}")

    # Load previous results if resuming
    processed_data = {}
    if cfg.test.resume:
        logger.info(f"Resuming from {cfg.test.resume}")
        try:
            with open(cfg.test.resume, encoding="utf-8") as f:
                for line in f:
                    result = json.loads(line)
                    processed_data[result["id"]] = result
            logger.info(f"Loaded {len(processed_data)} processed samples")
        except Exception as e:
            logger.error(f"Could not resume: {e}")

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
                    # Prevent full crash if a single sample fails
                    logger.error(f"Error at sample {i} (id={sample.get('id')}): {e}")
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

            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

    logger.info(f"Predictions saved to {prediction_path}")

    # Evaluation
    try:
        evaluator = instantiate(cfg.qa_evaluator, prediction_file=prediction_path)
        metrics = evaluator.evaluate()
        query_utils.print_metrics(metrics, logger)
    except Exception as e:
        logger.error(f"QA evaluation error: {e}")

    try:
        retrieval_evaluator = RetrievalEvaluator(prediction_file=prediction_path)
        retrieval_metrics = retrieval_evaluator.evaluate()
        query_utils.print_metrics(retrieval_metrics, logger)
    except Exception as e:
        logger.error(f"Retrieval evaluation error: {e}")


if __name__ == "__main__":
    main()
