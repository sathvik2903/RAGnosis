"""Vision parsing and provider orchestration tests (no network)."""

from pathlib import Path

import pytest
from PIL import Image

from multimodal.config import MultimodalConfig
from multimodal.schemas import ImageObservation
from multimodal.vision import (
    VisionProvider,
    observed_terms,
    parse_observations,
)


def _make_image(path: Path) -> Path:
    Image.new("RGB", (32, 32), "gray").save(path)
    return path


def test_parse_clean_json():
    body = {
        "observations": [
            {
                "label": "opacity",
                "description": "area of increased density",
                "confidence": 0.3,
                "location": "upper left",
                "caveat": "nonspecific",
            }
        ]
    }
    obs = parse_observations(body)
    assert len(obs) == 1
    assert obs[0].label == "opacity"
    assert obs[0].confidence == pytest.approx(0.3)


def test_parse_json_string():
    obs = parse_observations('{"observations": [{"label": "x", "description": "y"}]}')
    assert obs[0].label == "x"


def test_parse_json_wrapped_in_prose_and_fences():
    content = 'Here is the result:\n```json\n{"observations": [{"label": "z", "description": "d"}]}\n```'
    obs = parse_observations(content)
    assert obs[0].label == "z"


def test_parse_malformed_returns_empty():
    assert parse_observations("not json at all") == []
    assert parse_observations("") == []
    assert parse_observations(None) == []


def test_parse_observations_not_a_list():
    assert parse_observations({"observations": "oops"}) == []


def test_parse_skips_non_dict_entries():
    body = {"observations": ["bad", {"label": "ok", "description": "d"}]}
    obs = parse_observations(body)
    assert len(obs) == 1
    assert obs[0].label == "ok"


def test_parse_defaults_missing_label():
    obs = parse_observations({"observations": [{"description": "d"}]})
    assert obs[0].label == "unspecified"


def test_observed_terms_excludes_unspecified():
    obs = [
        ImageObservation("opacity", "d"),
        ImageObservation("unspecified", "d"),
    ]
    assert observed_terms(obs) == ["opacity"]


def test_vision_provider_not_configured_raises(tmp_path: Path):
    path = _make_image(tmp_path / "img.png")
    provider = VisionProvider(MultimodalConfig(openai_api_key=""))
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        provider.observe(path, "what is this")


def test_vision_provider_observe_with_injected_poster(tmp_path: Path):
    path = _make_image(tmp_path / "img.png")
    captured = {}

    def fake_poster(url, headers, payload, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = payload
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"observations": [{"label": "lesion", "description": "d", "confidence": 0.2}]}'
                    }
                }
            ]
        }

    provider = VisionProvider(
        MultimodalConfig(openai_api_key="test-key", vision_model="test-model"),
        poster=fake_poster,
    )
    observations, model = provider.observe(path, "describe features")
    assert model == "test-model"
    assert observations[0].label == "lesion"
    assert observations[0].confidence == pytest.approx(0.2)
    assert captured["url"].endswith("/chat/completions")
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    # The image must be transported as an image_url data URL.
    user_msg = captured["payload"]["messages"][-1]["content"]
    assert any(part.get("type") == "image_url" for part in user_msg)


def test_vision_provider_bad_response_shape_raises(tmp_path: Path):
    path = _make_image(tmp_path / "img.png")

    def bad_poster(url, headers, payload, timeout):
        return {"unexpected": True}

    provider = VisionProvider(
        MultimodalConfig(openai_api_key="k"), poster=bad_poster
    )
    with pytest.raises(RuntimeError, match="unexpected response shape"):
        provider.observe(path, "q")
