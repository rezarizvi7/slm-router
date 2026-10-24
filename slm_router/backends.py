"""Model backends behind a single normalized interface.

The router speaks the OpenAI Chat Completions shape on the outside (that's what most
dev tools emit), so the OpenAI-compatible backend is near pass-through and the
Anthropic backend translates to/from the Messages API.
"""

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass

import anthropic
import httpx

from .complexity import message_text

# Request fields we forward to OpenAI-compatible servers.
OPENAI_PASSTHROUGH = ("temperature", "top_p", "max_tokens", "max_completion_tokens", "stop", "seed",
                      "response_format", "tools", "tool_choice", "presence_penalty", "frequency_penalty")


@dataclass
class ChatResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    finish_reason: str = "stop"
    tool_calls: list[dict] | None = None
    usage_estimated: bool = False


@dataclass
class StreamEvent:
    """Either a text delta, or (last event) the final accumulated result."""
    delta: str | None = None
    final: ChatResult | None = None


class BackendError(Exception):
    pass


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class OpenAICompatBackend:
    """Ollama (/v1), vLLM, llama.cpp server, OpenAI, OpenRouter, ..."""

    def __init__(self, base_url: str, model: str, api_key: str, timeout_s: float,
                 client: httpx.AsyncClient | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self._client = client or httpx.AsyncClient()
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def _body(self, req: dict, stream: bool) -> dict:
        body = {"model": self.model, "messages": req["messages"], "stream": stream}
        body.update({k: req[k] for k in OPENAI_PASSTHROUGH if req.get(k) is not None})
        if stream:
            body["stream_options"] = {"include_usage": True}
        return body

    async def complete(self, req: dict) -> ChatResult:
        try:
            r = await self._client.post(f"{self.base_url}/chat/completions", json=self._body(req, False),
                                        headers=self._headers, timeout=self.timeout_s)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise BackendError(f"{self.model}: {e!r}") from e
        data = r.json()
        choice = data["choices"][0]
        msg = choice.get("message") or {}
        text = msg.get("content") or ""
        usage = data.get("usage") or {}
        return ChatResult(
            text=text,
            model=data.get("model", self.model),
            input_tokens=usage.get("prompt_tokens") or _approx_tokens(json.dumps(req["messages"])),
            output_tokens=usage.get("completion_tokens") or _approx_tokens(text),
            finish_reason=choice.get("finish_reason") or "stop",
            tool_calls=msg.get("tool_calls"),
            usage_estimated=not usage,
        )

    async def stream(self, req: dict) -> AsyncIterator[StreamEvent]:
        parts: list[str] = []
        usage: dict = {}
        finish = "stop"
        model = self.model
        try:
            async with self._client.stream("POST", f"{self.base_url}/chat/completions",
                                           json=self._body(req, True), headers=self._headers,
                                           timeout=self.timeout_s) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    chunk = json.loads(payload)
                    model = chunk.get("model", model)
                    usage = chunk.get("usage") or usage
                    for ch in chunk.get("choices", []):
                        d = (ch.get("delta") or {}).get("content")
                        if d:
                            parts.append(d)
                            yield StreamEvent(delta=d)
                        finish = ch.get("finish_reason") or finish
        except httpx.HTTPError as e:
            raise BackendError(f"{self.model}: {e!r}") from e
        text = "".join(parts)
        yield StreamEvent(final=ChatResult(
            text=text, model=model,
            input_tokens=usage.get("prompt_tokens") or _approx_tokens(json.dumps(req["messages"])),
            output_tokens=usage.get("completion_tokens") or _approx_tokens(text),
            finish_reason=finish, usage_estimated=not usage,
        ))


# ---------------------------------------------------------------------------
# Anthropic: OpenAI request shape -> Messages API, and back.
# ---------------------------------------------------------------------------

STOP_REASON_MAP = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length",
                   "tool_use": "tool_calls", "refusal": "content_filter", "pause_turn": "stop"}


def _image_block(url: str) -> dict:
    if url.startswith("data:"):
        header, data = url.split(",", 1)
        media_type = header[5:].split(";")[0]
        return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}
    return {"type": "image", "source": {"type": "url", "url": url}}


def _content_blocks(content) -> list[dict]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    blocks = []
    for p in content or []:
        if p.get("type") == "text":
            blocks.append({"type": "text", "text": p["text"]})
        elif p.get("type") == "image_url":
            url = p["image_url"]["url"] if isinstance(p["image_url"], dict) else p["image_url"]
            blocks.append(_image_block(url))
    return blocks


def to_anthropic(req: dict) -> tuple[str | None, list[dict], list[dict] | None]:
    """Returns (system, messages, tools) for the Messages API."""
    system_parts: list[str] = []
    out: list[dict] = []

    def push(role: str, blocks: list[dict]):
        # Messages API wants alternating roles; merge consecutive same-role turns.
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": blocks})

    for m in req["messages"]:
        role = m.get("role")
        if role in ("system", "developer"):
            system_parts.append(message_text(m.get("content")))
        elif role == "user":
            push("user", _content_blocks(m.get("content")))
        elif role == "assistant":
            blocks = _content_blocks(m.get("content"))
            for tc in m.get("tool_calls") or []:
                fn = tc["function"]
                blocks.append({"type": "tool_use", "id": tc["id"], "name": fn["name"],
                               "input": json.loads(fn.get("arguments") or "{}")})
            push("assistant", blocks)
        elif role == "tool":
            push("user", [{"type": "tool_result", "tool_use_id": m["tool_call_id"],
                           "content": message_text(m.get("content"))}])

    tools = None
    if req.get("tools"):
        tools = [{"name": t["function"]["name"],
                  "description": t["function"].get("description", ""),
                  "input_schema": t["function"].get("parameters") or {"type": "object", "properties": {}}}
                 for t in req["tools"] if t.get("type") == "function"]
    return ("\n\n".join(p for p in system_parts if p) or None), out, tools


def _tool_choice(tc) -> dict | None:
    # Current Claude models reject forced tool_choice ("any"/specific tool) with a 400,
    # so "required"/named choices degrade to "auto".
    if tc is None:
        return None
    if tc == "none":
        return {"type": "none"}
    return {"type": "auto"}


def from_anthropic(msg) -> ChatResult:
    text = "".join(b.text for b in msg.content if b.type == "text")
    tool_calls = [{"id": b.id, "type": "function",
                   "function": {"name": b.name, "arguments": json.dumps(b.input)}}
                  for b in msg.content if b.type == "tool_use"] or None
    return ChatResult(text=text, model=msg.model, input_tokens=msg.usage.input_tokens,
                      output_tokens=msg.usage.output_tokens,
                      finish_reason=STOP_REASON_MAP.get(msg.stop_reason, "stop"), tool_calls=tool_calls)


# The SDK raises TypeError (not APIError) when no credentials resolve; treat it as an
# upstream failure so the router can degrade instead of returning a 500.
_SDK_ERRORS = (anthropic.APIError, TypeError)


class AnthropicBackend:
    def __init__(self, model: str, api_key: str, timeout_s: float, max_tokens: int, effort: str,
                 refusal_fallback: bool, client: anthropic.AsyncAnthropic | None = None):
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.refusal_fallback = refusal_fallback
        self._client = client or anthropic.AsyncAnthropic(api_key=api_key or None, timeout=timeout_s)

    def _params(self, req: dict) -> dict:
        system, messages, tools = to_anthropic(req)
        p: dict = {
            "model": self.model,
            "max_tokens": req.get("max_completion_tokens") or req.get("max_tokens") or self.max_tokens,
            "messages": messages,
            "output_config": {"effort": self.effort},
        }
        # temperature/top_p are deliberately not forwarded: current Claude models reject them.
        if system:
            p["system"] = system
        if tools:
            p["tools"] = tools
            if tc := _tool_choice(req.get("tool_choice")):
                p["tool_choice"] = tc
        if stop := req.get("stop"):
            p["stop_sequences"] = [stop] if isinstance(stop, str) else stop
        if self.refusal_fallback:
            # Server-side refusal fallback: if the safety classifier declines, the API
            # re-runs the request on Anthropic's recommended fallback model in the same call.
            p["betas"] = ["server-side-fallback-2026-07-01"]
            p["fallbacks"] = "default"
        return p

    async def complete(self, req: dict) -> ChatResult:
        try:
            msg = await self._client.beta.messages.create(**self._params(req))
        except _SDK_ERRORS as e:
            raise BackendError(f"{self.model}: {e!r}") from e
        return from_anthropic(msg)

    async def stream(self, req: dict) -> AsyncIterator[StreamEvent]:
        try:
            async with self._client.beta.messages.stream(**self._params(req)) as s:
                async for text in s.text_stream:
                    yield StreamEvent(delta=text)
                final = await s.get_final_message()
        except _SDK_ERRORS as e:
            raise BackendError(f"{self.model}: {e!r}") from e
        yield StreamEvent(final=from_anthropic(final))

