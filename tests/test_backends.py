import json

import httpx
import pytest
import respx

from slm_router.backends import AnthropicBackend, BackendError, OpenAICompatBackend, to_anthropic


@respx.mock
async def test_openai_compat_complete():
    respx.post("http://local/v1/chat/completions").mock(return_value=httpx.Response(200, json={
        "model": "qwen", "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 2}}))
    b = OpenAICompatBackend("http://local/v1", "qwen", "k", 5)
    r = await b.complete({"messages": [{"role": "user", "content": "yo"}], "temperature": 0.2, "junk": 1})
    assert (r.text, r.input_tokens, r.output_tokens) == ("hi", 7, 2)
    sent = json.loads(respx.calls.last.request.content)
    assert sent["temperature"] == 0.2 and "junk" not in sent and sent["model"] == "qwen"


@respx.mock
async def test_openai_compat_stream_with_usage():
    sse = "".join(f"data: {json.dumps(c)}\n\n" for c in [
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}},
    ]) + "data: [DONE]\n\n"
    respx.post("http://local/v1/chat/completions").mock(return_value=httpx.Response(200, text=sse))
    b = OpenAICompatBackend("http://local/v1", "qwen", "", 5)
    evs = [e async for e in b.stream({"messages": [{"role": "user", "content": "x"}]})]
    assert "".join(e.delta for e in evs if e.delta) == "Hello"
    assert evs[-1].final.output_tokens == 2 and not evs[-1].final.usage_estimated


def test_to_anthropic_translation():
    req = {
        "messages": [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "wx", "arguments": '{"city":"Oslo"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "5C"},
            {"role": "user", "content": [{"type": "text", "text": "and this?"},
                                         {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]},
        ],
        "tools": [{"type": "function", "function": {"name": "wx", "parameters": {"type": "object"}}}],
    }
    system, msgs, tools = to_anthropic(req)
    assert system == "be terse"
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]  # tool result + next user merged
    assert msgs[1]["content"][0] == {"type": "tool_use", "id": "c1", "name": "wx", "input": {"city": "Oslo"}}
    assert msgs[2]["content"][0]["type"] == "tool_result"
    assert msgs[2]["content"][2]["source"] == {"type": "base64", "media_type": "image/png", "data": "AAAA"}
    assert tools == [{"name": "wx", "description": "", "input_schema": {"type": "object"}}]


async def test_anthropic_missing_credentials_is_backend_error(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    b = AnthropicBackend("claude-opus-5-5", "", 5, 100, "low", True)
    b._client.api_key = b._client.auth_token = None
    with pytest.raises(BackendError):
        await b.complete({"messages": [{"role": "user", "content": "hi"}]})
