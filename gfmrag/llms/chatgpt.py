import logging
import os
from typing import Union

from openai import OpenAI
from .base_language_model import BaseLanguageModel

logger = logging.getLogger(__name__)


class ChatGPT(BaseLanguageModel):
    def __init__(self, model_name_or_path: str, retry: int = 5, base_url: str = None):
        self.retry = retry
        self.model_name = model_name_or_path
        self.maximun_token = self._get_token_limit(self.model_name)

        # Token tracking variables
        self.total_tokens = 0
        self.total_requests = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

        # Fix environment variable access
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        client = OpenAI(
            api_key=api_key,
            base_url=base_url or "https://api.openai.com/v1",
        )
        self.client = client

    def _get_token_limit(self, model_name: str) -> int:
        """Get token limit for different models"""
        if "gpt-4" in model_name:
            return 8192 if "32k" in model_name else 4096
        elif "gpt-3.5" in model_name:
            return 4096
        else:
            return 4096

    def token_len(self, text: str) -> int:
        """Simple token estimation (rough approximation)"""
        return len(text.split()) * 1.3  # Rough estimate

    def generate_sentence(
            self, llm_input: str | list, system_input: str = ""
    ) -> str | Exception:
        """Generate sentence with token tracking"""
        try:
            # Format messages
            if isinstance(llm_input, list):
                messages = llm_input
            else:
                messages = []
                if system_input:
                    messages.append({"role": "system", "content": system_input})
                messages.append({"role": "user", "content": llm_input})

                # Call API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0,
                max_tokens=self.maximun_token,
            )

            # Extract content and usage
            content = response.choices[0].message.content

            # Track token usage
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                prompt_t = usage.prompt_tokens
                completion_t = usage.completion_tokens
                total_t = usage.total_tokens

                # Update counters
                self.total_requests += 1
                self.total_tokens += total_t
                self.prompt_tokens += prompt_t
                self.completion_tokens += completion_t

                # Calculate averages
                avg_total = self.total_tokens / self.total_requests
                avg_prompt = self.prompt_tokens / self.total_requests
                avg_completion = self.completion_tokens / self.total_requests

                # Log detailed usage
                logger.info(f"Token Usage - Prompt: {prompt_t}, "
                            f"Completion: {completion_t}, "
                            f"Total: {total_t}")
                logger.info(f"Cumulative - Total Requests: {self.total_requests}, "
                            f"Total Tokens: {self.total_tokens}, "
                            f"Average Total: {avg_total:.2f}, "
                            f"Average Prompt: {avg_prompt:.2f}, "
                            f"Average Completion: {avg_completion:.2f}")

            return content.strip() if content else ""

        except Exception as e:
            logger.error(f"Error in generate_sentence: {e}")
            return e

    def get_token_stats(self) -> dict:
        """Get current token usage statistics"""
        if self.total_requests == 0:
            return {
                "total_requests": 0,
                "total_tokens": 0,
                "average_tokens_per_request": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0
            }

        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "average_tokens_per_request": self.total_tokens / self.total_requests,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "average_prompt_tokens": self.prompt_tokens / self.total_requests,
            "average_completion_tokens": self.completion_tokens / self.total_requests
        }

    def reset_token_stats(self):
        """Reset token usage counters"""
        self.total_tokens = 0
        self.total_requests = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        logger.info("Token usage statistics reset")