import os
import time
from typing import List, Optional

import openai
from dotenv import load_dotenv

IGNORE_STR = "Wait"


class OpenAIUtils:
    def __init__(self, api_key: str, base_url: str, model_name: str, max_tokens: int, thinking_temperature: float = 0.0,
                 answer_temperature: float = 0.0, **kwargs):
        load_dotenv()
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.max_tokens_thinking = max_tokens
        self.thinking_temperature = thinking_temperature
        self.answer_temperature = answer_temperature
        self.kwargs = kwargs

        # Load sleep time from environment variable
        self.sleep_time = float(os.getenv('OPENAI_RETRY_SLEEP', 60))  # Default to 60 seconds if not set

    def generate_thinking(self, messages: List[dict], think_time_scale=1) -> str:
        """Generates intermediate thinking response."""
        assert think_time_scale > 0, "Think time scale must be greater than 0."

        thinking_messages = messages + [{"role": "assistant", "content": "Let me think about this...", "prefix": True}]

        stop_token = ["Final Answer:", "\\boxed", "Conclusion:"]
        thinking_content = self._get_response(
            thinking_messages,
            stop_token,
            temperature=self.thinking_temperature,
            max_tokens=self.max_tokens_thinking
        )

        print(f"Thinking content: {thinking_content}")

        max_thinking_tokens_tmp = self.max_tokens_thinking
        if max_thinking_tokens_tmp > 0:
            for _ in range(think_time_scale):
                max_thinking_tokens_tmp -= len(thinking_content)

                thinking_messages[-1]["content"] += f" {thinking_content} {IGNORE_STR}"
                thinking_content = self._get_response(
                    thinking_messages,
                    stop_token,
                    temperature=self.thinking_temperature,
                    max_tokens=max_thinking_tokens_tmp
                )

        return thinking_messages[-1]["content"] + thinking_content

    def generate_final_answer(self, messages: List[dict], thinking_content: str, max_attempts: int = 5) \
            -> (Optional)[str]:
        """Generates final answer based on the thinking content."""
        final_messages = messages + [
            {"role": "assistant", "content": f"{thinking_content}\n\nFinal Answer:", "prefix": True}]

        for _ in range(max_attempts):
            try:
                return self._get_response(
                    final_messages,
                    temperature=self.answer_temperature,
                    max_tokens=self.max_tokens_thinking
                )
            except Exception as e:
                print(f"Exception: {str(e)}")
                time.sleep(self.sleep_time)
        return None

    def _get_response(self, messages: List[dict], stop_token: Optional[List[str]] = None,
                      temperature: float = 0.0, max_tokens: int = 2048) -> str:
        """Handles API call to OpenAI chat completion."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop_token if stop_token else None
        )
        return response.choices[0].message.content

    def process_prompt(self, prompt: str, think_time_scale: int = 1) -> (str, str):
        """Processes a single prompt and returns the response."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant that completes sentences."},
            {"role": "user", "content": prompt}
        ]
        thinking_content = self.generate_thinking(messages, think_time_scale)
        return thinking_content, self.generate_final_answer(messages, thinking_content) or "Error generating response."
