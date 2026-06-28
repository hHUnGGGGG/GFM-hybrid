# Adapt from: https://github.com/OSU-NLP-Group/HippoRAG/blob/main/src/named_entity_extraction_parallel.py
import logging
from typing import Literal

from langchain_community.chat_models import ChatLlamaCpp, ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from gfmrag_hybrid.kg_construction.langchain_util import init_langchain_model
from gfmrag_hybrid.kg_construction.utils import extract_json_dict, processing_phrases

from .base_model import BaseNERModel

logger = logging.getLogger(__name__)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

# ── SYSTEM ──────────────────────────────────────────────────────────
system_prompt = (
    "Bạn là hệ thống trích xuất thực thể y tế tiếng Việt. "
    "Nhiệm vụ: nhận diện và trích xuất các thực thể quan trọng từ câu hỏi "
    "về dược phẩm và y tế. "
    "Các loại thực thể cần nhận diện: "
    "tên thuốc (vd: Biviantac, Captopril), "
    "hoạt chất (vd: atorvastatin, amoxicillin), "
    "bệnh lý (vd: tăng huyết áp, loét dạ dày), "
    "triệu chứng (vd: trướng bụng, đầy hơi, buồn nôn), "
    "tác dụng phụ (vd: chóng mặt, phát ban), "
    "đối tượng bệnh nhân (vd: phụ nữ có thai, trẻ em, người cao tuổi), "
    "chống chỉ định (vd: suy thận, dị ứng penicillin), "
    "tương tác thuốc (vd: warfarin, rượu bia), "
    "đường dùng (vd: đường uống, tiêm tĩnh mạch), "
    "liều dùng (vd: 500mg, 2 lần/ngày). "
    "Chỉ trích xuất thực thể có trong câu hỏi, không thêm thực thể ngoài. "
    "Trả về JSON với key 'named_entities'."
)

# ── FEW-SHOT EXAMPLES ──────────────────────────────────────────────────
# Example 1: chỉ định / triệu chứng
query_prompt_one_shot_input = """Hãy trích xuất tất cả các thực thể quan trọng từ câu hỏi y tế dưới đây.
Trả về kết quả dạng JSON.

Câu hỏi: Biviantac có thể điều trị trướng bụng, đầy hơi không?

"""
query_prompt_one_shot_output = """
{"named_entities": ["Biviantac", "trướng bụng", "đầy hơi"]}
"""

# Example 2: chống chỉ định theo đối tượng
query_prompt_one_shot_input_2 = """Câu hỏi: Thuốc Captopril có dùng được cho phụ nữ có thai và bệnh nhân suy thận không?

"""
query_prompt_one_shot_output_2 = """
{"named_entities": ["Captopril", "phụ nữ có thai", "suy thận"]}
"""

# Example 3: tác dụng phụ theo đối tượng
query_prompt_one_shot_input_3 = """Câu hỏi: Ceritine gây ra tác dụng phụ gì ở trẻ em dưới 6 tuổi?

"""
query_prompt_one_shot_output_3 = """
{"named_entities": ["Ceritine", "trẻ em dưới 6 tuổi"]}
"""

# Example 4: tương tác thuốc
query_prompt_one_shot_input_4 = """Câu hỏi: Bệnh nhân loét dạ dày có dùng Aspirin 81 cùng Ibuprofen được không?

"""
query_prompt_one_shot_output_4 = """
{"named_entities": ["loét dạ dày", "Aspirin 81", "Ibuprofen"]}
"""

# ── QUERY TEMPLATE ─────────────────────────────────────────────────────
query_prompt_template = """Câu hỏi: {}

"""


class LLMNERModelVietnamese(BaseNERModel):
    """A Named Entity Recognition (NER) model that uses Language Models (LLMs) for entity extraction.

    This class implements entity extraction using various LLM backends (OpenAI, Together, Ollama, llama.cpp)
    through the Langchain interface. It processes text input and returns a list of extracted named entities.

    Args:
        llm_api (Literal["openai", "nvidia", "together", "ollama", "llama.cpp"]): The LLM backend to use. Defaults to "openai".
        model_name (str): Name of the specific model to use. Defaults to "gpt-4o-mini".
        max_tokens (int): Maximum number of tokens in the response. Defaults to 1024.

    Methods:
        __call__: Extracts named entities from the input text.

    Raises:
        Exception: If there's an error in extracting or processing named entities.
    """

    def __init__(
        self,
        llm_api: Literal[
            "openai", "nvidia", "together", "ollama", "llama.cpp"
        ] = "openai",
        model_name: str = "gpt-4o-mini",
        max_tokens: int = 1024,
    ):
        self.llm_api = llm_api
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.client = init_langchain_model(llm_api, model_name)

    def __call__(self, text: str) -> list:
        """Process text input to extract named entities using different chat models.

        Args:
            text (str): The input text to extract named entities from.

        Returns:
            list: A list of processed named entities extracted from the text.
                 Returns empty list if extraction fails.
        """
        query_ner_prompts = ChatPromptTemplate.from_messages(
            [
                SystemMessage(system_prompt),
                HumanMessage(query_prompt_one_shot_input),
                AIMessage(query_prompt_one_shot_output),
                HumanMessage(query_prompt_one_shot_input_2),
                AIMessage(query_prompt_one_shot_output_2),
                HumanMessage(query_prompt_one_shot_input_3),
                AIMessage(query_prompt_one_shot_output_3),
                HumanMessage(query_prompt_one_shot_input_4),
                AIMessage(query_prompt_one_shot_output_4),
                HumanMessage(query_prompt_template.format(text)),
            ]
        )
        query_ner_messages = query_ner_prompts.format_prompt()
        logger.debug(f"Query ner Prompt: {query_ner_messages.to_messages()}")
        json_mode = False
        if isinstance(self.client, ChatOpenAI):  # JSON mode
            chat_completion = self.client.invoke(
                query_ner_messages.to_messages(),
                temperature=0,
                max_tokens=self.max_tokens,
                stop=["\n\n"],
                response_format={"type": "json_object"},
            )
            response_content = chat_completion.content
            logger.debug(f"Query NER Prompt: {response_content}")
            chat_completion.response_metadata["token_usage"]["total_tokens"]
            json_mode = True
        elif isinstance(self.client, ChatOllama) or isinstance(
            self.client, ChatLlamaCpp
        ):
            response_content = self.client.invoke(query_ner_messages.to_messages())
            if hasattr(response_content, "content"):
                response_content = response_content.content
            response_content = extract_json_dict(response_content)
        else:  # no JSON mode
            chat_completion = self.client.invoke(
                query_ner_messages.to_messages(),
                temperature=0,
                max_tokens=self.max_tokens,
                stop=["\n\n"],
            )
            response_content = chat_completion.content
            response_content = extract_json_dict(response_content)
            chat_completion.response_metadata["token_usage"]["total_tokens"]

        if not json_mode:
            try:
                assert "named_entities" in response_content
                response_content = str(response_content)
            except Exception as e:
                print("Query NER exception", e)
                response_content = {"named_entities": []}

        try:
            ner_list = eval(response_content)["named_entities"]
            query_ner_list = [processing_phrases(ner) for ner in ner_list]
            return query_ner_list
        except Exception as e:
            logger.error(f"Error in extracting named entities: {e}")
            return []