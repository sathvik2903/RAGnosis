"""CohereGenerator (Chat API V2) primary/fallback and parsing tests (no network).

These exercise the V2 integration via an injected fake ``ClientV2``: the fake
accepts ``chat(model=, messages=, temperature=)`` and returns a response shaped
like the real SDK (``response.message.content[0].text``), so the generator's
V2 request format and response parsing are both covered offline.
"""

import pytest

from multimodal.config import MultimodalConfig
from multimodal.errors import NotConfiguredError
from multimodal.service import CohereGenerator


class _ContentBlock:
    def __init__(self, text):
        self.text = text


class _Message:
    def __init__(self, blocks):
        self.content = blocks


class _V2Resp:
    """Mirrors the Cohere V2 chat response: ``.message.content[0].text``."""

    def __init__(self, blocks):
        self.message = _Message(blocks)


class _FakeClientV2:
    """Records calls and returns/raises per-model behaviour (V2 signature)."""

    def __init__(self, behaviour):
        # behaviour: dict model -> ("text", value) | ("blocks", list) | ("raise", exc)
        self.behaviour = behaviour
        self.calls = []
        self.last_messages = None
        self.last_temperature = None

    def chat(self, *, model, messages, temperature):
        self.calls.append(model)
        self.last_messages = messages
        self.last_temperature = temperature
        kind, value = self.behaviour[model]
        if kind == "raise":
            raise value
        if kind == "blocks":
            return _V2Resp(value)
        # "text": single content block containing the string
        return _V2Resp([_ContentBlock(value)])


def _generator(behaviour, config=None):
    gen = CohereGenerator(
        config
        or MultimodalConfig(
            cohere_api_key="k",
            cohere_model="primary-model",
            cohere_fallback_model="fallback-model",
        )
    )
    gen._client = _FakeClientV2(behaviour)  # inject V2 fake
    return gen


def test_not_configured_raises():
    gen = CohereGenerator(MultimodalConfig(cohere_api_key=""))
    with pytest.raises(NotConfiguredError):
        gen.generate("prompt")


def test_primary_success():
    gen = _generator({"primary-model": ("text", "a good answer")})
    text, model = gen.generate("prompt")
    assert text == "a good answer"
    assert model == "primary-model"


def test_v2_request_uses_messages_format():
    # The generator must send the V2 ``messages=[{role, content}]`` shape.
    gen = _generator({"primary-model": ("text", "ok")})
    gen.generate("hello world")
    assert gen._client.last_messages == [{"role": "user", "content": "hello world"}]
    assert gen._client.last_temperature == 0.1


def test_v2_response_parsing_concatenates_blocks():
    # ``message.content`` may contain multiple text blocks; parse them in order.
    gen = _generator(
        {"primary-model": ("blocks", [_ContentBlock("Hello, "), _ContentBlock("world.")])}
    )
    text, model = gen.generate("prompt")
    assert text == "Hello, world."
    assert model == "primary-model"


def test_v2_response_parsing_dict_block():
    # Defensive: a dict-shaped content block is also read.
    gen = _generator({"primary-model": ("blocks", [{"text": "dict answer"}])})
    text, _ = gen.generate("prompt")
    assert text == "dict answer"


def test_empty_primary_response_raises():
    gen = _generator({"primary-model": ("text", "   ")})
    with pytest.raises(RuntimeError, match="empty"):
        gen.generate("prompt")


def test_empty_content_list_raises():
    # No content blocks at all is treated as an empty response, not a crash.
    gen = _generator({"primary-model": ("blocks", [])})
    with pytest.raises(RuntimeError, match="empty"):
        gen.generate("prompt")


def test_fallback_on_model_error():
    gen = _generator(
        {
            "primary-model": ("raise", Exception("model not found: decommissioned")),
            "fallback-model": ("text", "fallback answer"),
        }
    )
    text, model = gen.generate("prompt")
    assert text == "fallback answer"
    assert model == "fallback-model"
    assert gen._client.calls == ["primary-model", "fallback-model"]


def test_fallback_on_v2_unsupported_model_error():
    # The real V2 migration was triggered by an "unsupported model" style error;
    # such a message must be treated as a model-availability problem -> fallback.
    gen = _generator(
        {
            "primary-model": (
                "raise",
                Exception(
                    "invalid request: this model is not supported with '/v1/chat'"
                ),
            ),
            "fallback-model": ("text", "fallback answer"),
        }
    )
    text, model = gen.generate("prompt")
    assert text == "fallback answer"
    assert model == "fallback-model"


def test_primary_success_does_not_call_fallback():
    gen = _generator(
        {
            "primary-model": ("text", "primary answer"),
            "fallback-model": ("text", "should not be used"),
        }
    )
    text, model = gen.generate("prompt")
    assert text == "primary answer"
    assert model == "primary-model"
    assert gen._client.calls == ["primary-model"]


def test_non_model_error_does_not_trigger_fallback():
    # A generic network/timeout error is not a model-availability problem, so we
    # must NOT silently retry on the fallback model (avoids masking real errors).
    gen = _generator(
        {"primary-model": ("raise", Exception("connection timed out"))}
    )
    with pytest.raises(Exception, match="timed out"):
        gen.generate("prompt")
    assert gen._client.calls == ["primary-model"]


def test_fallback_also_fails_propagates():
    gen = _generator(
        {
            "primary-model": ("raise", Exception("unknown model")),
            "fallback-model": ("raise", Exception("unknown model too")),
        }
    )
    with pytest.raises(Exception, match="unknown model too"):
        gen.generate("prompt")


def test_no_infinite_retry():
    # Each model is attempted at most once; total attempts bounded by 2.
    gen = _generator(
        {
            "primary-model": ("raise", Exception("model decommissioned")),
            "fallback-model": ("text", "ok"),
        }
    )
    gen.generate("prompt")
    assert len(gen._client.calls) == 2
