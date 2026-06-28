"""
LLM-as-Judge cho ViMedQA — sử dụng OpenAI SDK với custom base_url (Yescale)
=========================================================
Cách dùng:
    export OPENAI_API_KEY="sk-..."
    python llm_judge_vimedqa_yescale.py --input prediction.jsonl --output evaluated.jsonl
    python llm_judge_vimedqa_yescale.py --input prediction.jsonl --output evaluated.jsonl --workers 5
    python llm_judge_vimedqa_yescale.py --input prediction.jsonl --output evaluated.jsonl --resume
"""

import json
import os
import time
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai
from openai import OpenAI

# Đọc khoá từ file .env nếu có (tùy chọn, không bắt buộc)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL = "gpt-4o-mini"      # Đổi tên model nếu Yescale yêu cầu định dạng tên khác
MAX_TOKENS = 512
TEMPERATURE = 0.0          # Tính nhất quán cao khi chấm điểm
MAX_RETRIES = 3
RETRY_DELAY = 2            # giây, tăng dần (exponential backoff)
RATE_LIMIT_DELAY = 0.3     # giây giữa các request (workers=1)

# ── Judge prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "Bạn là chuyên gia y tế và giám khảo công bằng, đánh giá câu trả lời "
    "của mô hình AI trong lĩnh vực y dược tiếng Việt. "
    "Luôn trả về JSON hợp lệ."
)

JUDGE_PROMPT = """\
Nhiệm vụ: Đánh giá độ chính xác của câu trả lời do mô hình AI tạo ra \
so với đáp án tham chiếu cho câu hỏi y tế dưới đây.

[Tiêu chí chấm điểm — thang 1→5]
5 – Hoàn toàn chính xác, đầy đủ, bám sát đáp án tham chiếu, không có lỗi sai.
4 – Hầu hết chính xác, bao phủ các ý chính, sai sót nhỏ không ảnh hưởng nghĩa.
3 – Đúng một phần, thiếu chi tiết quan trọng hoặc có vài điểm không chính xác.
2 – Sai nhiều, bỏ sót thông tin quan trọng hoặc gây hiểu lầm.
1 – Hoàn toàn sai, không liên quan, hoặc từ chối trả lời khi có thể trả lời.

[Dữ liệu]
Question: {question}
Reference Answer: {reference}
Model Response: {response}

[Yêu cầu output]
Trả về JSON với đúng 2 trường:
{{"reasoning": "lý do ngắn gọn (≤60 từ, tiếng Việt)", "score": <số nguyên 1-5>}}
"""


# ── Core evaluation ───────────────────────────────────────────────────────────
def evaluate_single(
    client: OpenAI,
    question: str,
    reference: str,
    response: str,
    record_id: str = "",
) -> dict:
    """Gọi LLM để chấm 1 mẫu, có retry với exponential backoff."""
    prompt = JUDGE_PROMPT.format(
        question=question,
        reference=reference,
        response=response,
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}, # Cần đảm bảo Yescale hỗ trợ json_object
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            raw = completion.choices[0].message.content.strip()

            result = json.loads(raw)

            # Validate
            score = int(result.get("score", 0))
            if not (1 <= score <= 5):
                raise ValueError(f"Score ngoài phạm vi: {score}")

            return {
                "reasoning": str(result.get("reasoning", "")),
                "score": score,
            }

        except openai.RateLimitError:
            wait = RETRY_DELAY * (2 ** attempt)
            log.warning("Rate limit — chờ %ds (lần %d/%d) [%s]", wait, attempt, MAX_RETRIES, record_id)
            time.sleep(wait)

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            log.warning("Parse lỗi lần %d/%d [%s]: %s\nNội dung raw: %s", attempt, MAX_RETRIES, record_id, e, raw)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

        except Exception as e:
            log.error("Lỗi API [%s]: %s", record_id, e)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    log.error("Bỏ qua [%s] sau %d lần thử.", record_id, MAX_RETRIES)
    return {"reasoning": "Max retries exceeded", "score": 0}


# ── Load / Resume ─────────────────────────────────────────────────────────────
def load_already_evaluated(output_file: str) -> set:
    """Đọc file output đã có, trả về set ID đã được chấm."""
    done_ids = set()
    p = Path(output_file)
    if p.exists():
        with p.open(encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if "llm_judge_score" in obj and obj["llm_judge_score"] > 0:
                        done_ids.add(obj.get("id", ""))
                except json.JSONDecodeError:
                    pass
    return done_ids


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge cho ViMedQA sử dụng Yescale")
    parser.add_argument("--input",   default="prediction.jsonl",   help="File JSONL đầu vào")
    parser.add_argument("--output",  default="evaluated.jsonl",    help="File JSONL đầu ra")
    parser.add_argument("--workers", type=int, default=5,          help="Số luồng song song (mặc định 1)")
    parser.add_argument("--resume",  action="store_true",          help="Tiếp tục từ điểm dừng")
    parser.add_argument("--limit",   type=int, default=0,          help="Chỉ chấm N câu đầu (0 = tất cả)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Chưa set OPENAI_API_KEY. Hãy chạy: export OPENAI_API_KEY='sk-...'")

    # Khởi tạo client với base_url tùy chỉnh
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.yescale.io/v1"
    )

    # Đọc input
    with open(args.input, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if args.limit > 0:
        records = records[: args.limit]

    # Resume: lọc bỏ câu đã chấm
    done_ids = set()
    if args.resume:
        done_ids = load_already_evaluated(args.output)
        log.info("Resume: đã có %d câu, bỏ qua.", len(done_ids))

    todo = [r for r in records if r.get("id") not in done_ids]
    total = len(records)
    skip  = total - len(todo)
    log.info("Tổng: %d | Bỏ qua: %d | Cần chấm: %d | Workers: %d",
             total, skip, len(todo), args.workers)

    # Mở output (append nếu resume, ghi mới nếu không)
    open_mode = "a" if args.resume else "w"
    scores_all = []

    with open(args.output, open_mode, encoding="utf-8") as f_out:

        def process_record(data):
            rid = data.get("id", "?")

            # --- XỬ LÝ LẤY FINAL_ANSWER TỪ RESPONSE ---
            raw_response = data.get("response", "")
            final_answer_text = ""

            if isinstance(raw_response, dict):
                # Trường hợp response là một từ điển JSON
                final_answer_text = raw_response.get("final_answer", str(raw_response))
            elif isinstance(raw_response, str):
                # Thử parse JSON trong trường hợp nó được lưu dạng string
                try:
                    parsed_resp = json.loads(raw_response)
                    if isinstance(parsed_resp, dict):
                        final_answer_text = parsed_resp.get("final_answer", raw_response)
                    else:
                        final_answer_text = raw_response
                except json.JSONDecodeError:
                    final_answer_text = raw_response
            else:
                final_answer_text = str(raw_response)

            # Nếu final_answer trống hoặc null, để nó là chuỗi rỗng
            if not final_answer_text or final_answer_text == "null":
                final_answer_text = ""
            # ------------------------------------------

            result = evaluate_single(
                client,
                question=data.get("question", ""),
                reference=data.get("answer", ""),
                response=final_answer_text, # Chỉ truyền final_answer vào đây
                record_id=rid,
            )
            return data, result

        if args.workers == 1:
            # Sequential — dễ debug, tránh rate limit
            for i, data in enumerate(todo, 1):
                data, result = process_record(data)
                data["llm_judge_reasoning"] = result["reasoning"]
                data["llm_judge_score"]     = result["score"]
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                f_out.flush()
                scores_all.append(result["score"])
                log.info("[%d/%d] %s | score=%d | %s",
                         i + skip, total, data.get("id",""), result["score"],
                         result["reasoning"][:60])
                time.sleep(RATE_LIMIT_DELAY)
        else:
            # Parallel — nhanh hơn nhưng cẩn thận rate limit
            futures = {}
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                for data in todo:
                    f = executor.submit(process_record, data)
                    futures[f] = data

            done_count = skip
            for future in as_completed(futures):
                done_count += 1
                try:
                    data, result = future.result()
                    data["llm_judge_reasoning"] = result["reasoning"]
                    data["llm_judge_score"]     = result["score"]
                    f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                    f_out.flush()
                    scores_all.append(result["score"])
                    log.info("[%d/%d] %s | score=%d",
                             done_count, total, data.get("id",""), result["score"])
                except Exception as e:
                    log.error("Future lỗi: %s", e)

    # ── Thống kê ────────────────────────────────────────────────────────────
    valid = [s for s in scores_all if s > 0]
    if valid:
        avg  = sum(valid) / len(valid)
        pct  = (avg / 5.0) * 100
        dist = {s: valid.count(s) for s in range(1, 6)}

        print("\n" + "=" * 50)
        print(f"  Kết quả đánh giá LLM-as-Judge — ViMedQA (Yescale)")
        print("=" * 50)
        print(f"  Đã chấm:          {len(valid)} / {total} câu")
        print(f"  Điểm trung bình:  {avg:.3f} / 5.0")
        print(f"  Accuracy (÷5×100): {pct:.2f}%")
        print()
        print("  Phân bố điểm:")
        labels = {5: "Hoàn toàn đúng", 4: "Hầu hết đúng", 3: "Một phần đúng",
                  2: "Sai nhiều",       1: "Sai hoàn toàn"}
        for score in range(5, 0, -1):
            n   = dist[score]
            # Tránh chia cho 0 nếu max(dist.values()) = 0
            max_val = max(dist.values()) if max(dist.values()) > 0 else 1
            bar = "█" * int(n / max_val * 20)
            print(f"    {score} ({labels[score]:18s}): {n:4d}  {bar}")
        print()
        pass_n = dist.get(4, 0) + dist.get(5, 0)
        print(f"  PASS (score ≥ 4): {pass_n} câu  ({pass_n/len(valid)*100:.1f}%)")
        print(f"  FAIL (score ≤ 3): {len(valid)-pass_n} câu  ({(len(valid)-pass_n)/len(valid)*100:.1f}%)")
        print(f"\n  Kết quả lưu tại: {args.output}")
        print("=" * 50)
    else:
        log.warning("Không có điểm hợp lệ nào.")


if __name__ == "__main__":
    main()