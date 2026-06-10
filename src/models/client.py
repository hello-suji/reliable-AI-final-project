from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI


@dataclass
class ModelConfig:
    provider: str
    model_name: str
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout: int = 120
    base_url: Optional[str] = None
    api_key: Optional[str] = None


class LLMClient:
    def __init__(self, config: ModelConfig):
        self.config = config
        # 마지막 API 호출의 토큰 사용량 (누적 추적용)
        self.last_usage: dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        if config.provider == "local":
            self.client = OpenAI(
                base_url=config.base_url,
                api_key=config.api_key or "EMPTY",
            )
        elif config.provider == "openai":
            self.client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY")
            )
        else:
            raise ValueError(f"지원하지 않는 provider: {config.provider}")

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant",
        temperature: Optional[float] = None,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout,
        )
        # API 응답에서 실제 토큰 사용량 저장
        if response.usage:
            self.last_usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }
        return response.choices[0].message.content if response.choices else ""