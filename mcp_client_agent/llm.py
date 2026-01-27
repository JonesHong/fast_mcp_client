from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional, Protocol

import ollama
import openai


class ChatProvider(Protocol):
    async def complete_text(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str: ...

    async def stream_deltas(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[str]: ...


class OpenAIChatProvider:
    def __init__(self, *, api_key: str, base_url: Optional[str] = None) -> None:
        if base_url:
            self._client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        else:
            self._client = openai.AsyncOpenAI(api_key=api_key)

    async def complete_text(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        resp = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    async def stream_deltas(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for event in stream:
            delta: Any = None
            try:
                delta = event.choices[0].delta.content
            except Exception:
                delta = None
            if delta:
                yield delta


class OllamaChatProvider:
    def __init__(self, *, host: Optional[str] = None, keep_alive: float | str | None = None) -> None:
        # ollama python client expects host like "http://localhost:11434"
        self._client = ollama.AsyncClient(host=host) if host else ollama.AsyncClient()
        self._keep_alive = keep_alive

    async def complete_text(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        resp = await self._client.chat(
            model=model,
            messages=messages,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            keep_alive=self._keep_alive,
            stream=False,
        )
        msg = (resp or {}).get("message") or {}
        return str(msg.get("content") or "")

    async def stream_deltas(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        stream = await self._client.chat(
            model=model,
            messages=messages,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            keep_alive=self._keep_alive,
            stream=True,
        )
        async for chunk in stream:
            msg = (chunk or {}).get("message") or {}
            content = msg.get("content")
            if content:
                yield str(content)


@dataclass(frozen=True)
class LLMBackend:
    name: str  # "openai" | "ollama"
    provider: ChatProvider
    model: str


class LLMRouter:
    def __init__(self, *, primary: LLMBackend, fallback: Optional[LLMBackend] = None) -> None:
        self.primary = primary
        self.fallback = fallback

    @staticmethod
    def _normalize_provider(name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        n = str(name).strip().lower()
        if n in {"openai"}:
            return "openai"
        if n in {"ollama"}:
            return "ollama"
        return n

    def _pick_backend(self, provider_override: Optional[str]) -> LLMBackend:
        provider = self._normalize_provider(provider_override)
        if provider is None:
            return self.primary

        if provider == self.primary.name:
            return self.primary
        if self.fallback and provider == self.fallback.name:
            return self.fallback
        raise ValueError(f"Unknown/unsupported provider override: {provider_override}")

    async def chat_complete_text(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        backend = self._pick_backend(provider)
        chosen_model = model or backend.model
        try:
            return await backend.provider.complete_text(
                model=chosen_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as primary_exc:
            if not self.fallback or backend is self.fallback:
                raise
            fb = self.fallback
            # IMPORTANT: do not reuse `model` override across providers.
            # The override usually refers to the primary provider's model tag (e.g. an Ollama model),
            # and would be invalid if we fall back to OpenAI.
            return await fb.provider.complete_text(
                model=fb.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    async def chat_stream_deltas(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> AsyncIterator[str]:
        backend = self._pick_backend(provider)
        chosen_model = model or backend.model

        yielded = False
        try:
            async for d in backend.provider.stream_deltas(
                model=chosen_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                yielded = True
                yield d
            return
        except Exception as primary_exc:
            if yielded or not self.fallback or backend is self.fallback:
                raise
            fb = self.fallback
            async for d in fb.provider.stream_deltas(
                model=fb.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                yield d
