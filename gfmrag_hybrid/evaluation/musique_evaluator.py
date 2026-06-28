# Adapt from: https://github.com/OSU-NLP-Group/HippoRAG/blob/main/src/qa/musique_evaluation.py
import collections
import re
import string
from collections.abc import Callable

from gfmrag_hybrid.evaluation.base_evaluator import BaseEvaluator


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles, and extra whitespace."""

    def remove_articles(text: str) -> str:
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s: str) -> list:
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold: str, a_pred: str) -> int:
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold: str, a_pred: str) -> tuple:
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks), int(gold_toks == pred_toks), int(gold_toks == pred_toks)
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def metric_max_over_ground_truths(
    metric_fn: Callable, prediction: str, ground_truths: list
) -> float:
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(ground_truth, prediction)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def metric_max_f1_over_ground_truths(
    metric_fn: Callable, prediction: str, ground_truths: list
) -> tuple:
    max_f1, max_precision, max_recall = 0, 0, 0
    for ground_truth in ground_truths:
        f1, prec, recal = metric_fn(prediction, ground_truth)
        if f1 > max_f1:
            max_f1 = f1
            max_precision = prec
            max_recall = recal
    return max_f1, max_precision, max_recall


class MusiqueEvaluator(BaseEvaluator):
    """
    MusiqueEvaluator
    """

    def evaluate(self) -> dict:
        metrics = {"em": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

        for pred in self.data:
            # 1. Bảo vệ lỗi Null (Trường hợp LLM bỏ cuộc trả về null)
            raw_resp = pred.get("response", "")
            if raw_resp is None:
                pre_ans = ""
            else:
                # 2. Xử lý các tiền tố trả lời khác nhau của LLM
                raw_resp = str(raw_resp)
                if "So the answer is:" in raw_resp:
                    pre_ans = raw_resp.split("So the answer is:")[-1].strip()
                elif "Answer:" in raw_resp:
                    pre_ans = raw_resp.split("Answer:")[-1].strip()
                else:
                    pre_ans = raw_resp

            # 3. Lấy danh sách đáp án chuẩn (Gold answers)
            gold_answers = [pred.get("answer", "")] + pred.get("answer_aliases", [])

            # 4. Tính toán điểm số
            em = metric_max_over_ground_truths(compute_exact, pre_ans, gold_answers)
            (
                f1,
                precision,
                recall,
            ) = metric_max_f1_over_ground_truths(compute_f1, pre_ans, gold_answers)

            metrics["em"] += em
            metrics["f1"] += f1
            metrics["precision"] += precision
            metrics["recall"] += recall

        # Tính trung bình trên toàn bộ dataset
        total_samples = len(self.data)
        if total_samples > 0:
            metrics["em"] /= total_samples
            metrics["f1"] /= total_samples
            metrics["precision"] /= total_samples
            metrics["recall"] /= total_samples

        return metrics
