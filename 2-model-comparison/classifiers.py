"""Model backends: each Classifier sends chat messages to a model and returns its reply.

To add a model, subclass Classifier (implement classify) or reuse
TogetherClassifier, then add it to MODELS. prompt_style picks which prompt
format in classify_eval.py the model receives.
"""

import os
import re
from abc import ABC, abstractmethod
from pathlib import Path

NO_REASONING = {"reasoning": {"enabled": False}}


class Classifier(ABC):
    """Sends the classification prompt to a model and returns its raw reply."""

    prompt_style = "default"

    @abstractmethod
    def classify(self, messages: list[dict]) -> str:
        ...


class TogetherClassifier(Classifier):
    """A chat model served by Together AI (API key in .env as TOGETHER_AI_API_KEY).

    Always streams, since some models (e.g. Qwen3.8-Flash) only support
    streaming. Reasoning is turned off by default so models answer directly;
    models without reasoning accept and ignore the option.
    """

    def __init__(self, model, max_tokens=1024, prompt_style="default", extra_body=NO_REASONING):
        self.model = model
        self.max_tokens = max_tokens
        self.prompt_style = prompt_style
        self.extra_body = extra_body

    def classify(self, messages):
        stream = _together_client().chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            max_tokens=self.max_tokens,
            stream=True,
            extra_body=self.extra_body,
        )
        content, finish_reason = "", None
        for chunk in stream:
            if chunk.choices:
                content += chunk.choices[0].delta.content or ""
                finish_reason = chunk.choices[0].finish_reason or finish_reason
        if finish_reason == "length":
            raise RuntimeError(f"hit max_tokens={self.max_tokens} before answering: {content!r}")
        # Some models put their reasoning inline instead of in a separate field.
        return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)


_client = None


def _together_client():
    global _client
    if _client is None:
        from dotenv import load_dotenv
        from together import Together

        load_dotenv(Path(__file__).parent.parent / ".env")  # .env lives at the project root
        _client = Together(api_key=os.environ["TOGETHER_AI_API_KEY"], max_retries=5)
    return _client


MODELS = {
    "qwen3.8-flash": TogetherClassifier("Qwen/Qwen3.8-Flash"),
    "deepseek-v4.1-flash": TogetherClassifier("deepseek-ai/DeepSeek-V4.1-Flash"),
    "llama-3.3-70b": TogetherClassifier("meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    # Tev1 caps output at 8 tokens and is built to pick exactly one lettered
    # option, so it gets the single-choice "decision" prompt (see tev-example.py).
    "tev1-4b": TogetherClassifier(
        "together/Tev1-4B-experimental",
        max_tokens=8,
        prompt_style="decision",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    ),
    # Not usable as-is on Together:
    # - google/gemma-4-31B-it requires a dedicated endpoint (not serverless).
}
