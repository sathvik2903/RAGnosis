from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable

import requests

from .config import MultimodalConfig
from .errors import NotConfiguredError
from .image import encode_for_vision, inspect_image
from .safety import IMAGE_RULES, build_system_instruction
from .schemas import ImageObservation

# A poster takes (url, headers, json_payload, timeout) and returns the decoded
# JSON body. Injectable so the orchestration/parsing can be tested without network.
Poster = Callable[[str, dict, dict, float], dict]


def _default_poster(url: str, headers: dict, payload: dict, timeout: float) -> dict:
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(content: Any) -> dict:
    """Best-effort extraction of a JSON object from a model response.

    Handles clean JSON, JSON already parsed into a dict, and JSON wrapped in
    prose or ```json fences. Returns ``{}`` when nothing parseable is found so a
    malformed response degrades to "no observations" instead of crashing.
    """
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        return {}
    text = content.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    match = _JSON_OBJECT_RE.search(text)
    if match:
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def parse_observations(content: Any) -> list[ImageObservation]:
    """Convert a vision-model response body into validated observations.

    Pure function: no network, no file access. Silently drops malformed entries
    and clamps confidence (handled in ``ImageObservation``), so a partially bad
    response still yields the usable observations.
    """
    parsed = _extract_json(content)
    raw = parsed.get("observations")
    if not isinstance(raw, list):
        return []

    observations: list[ImageObservation] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "unspecified")).strip() or "unspecified"
        description = str(item.get("description", "")).strip()
        location = item.get("location")
        caveat = item.get("caveat")
        observations.append(
            ImageObservation(
                label=label,
                description=description,
                confidence=item.get("confidence"),
                location=str(location).strip() if location else None,
                caveat=str(caveat).strip() if caveat else None,
            )
        )
    return observations


class VisionProvider:
    """OpenAI-compatible vision adapter.

    The endpoint and model are configuration, not application logic, so RAGnosis
    stays independent of a single vendor and a local VLM adapter can be swapped
    in later without touching the service layer. The HTTP call is injectable to
    keep orchestration and parsing testable offline.
    """

    def __init__(
        self, config: MultimodalConfig | None = None, poster: Poster | None = None
    ) -> None:
        self.config = config or MultimodalConfig.from_env()
        self._poster = poster or _default_poster

    @property
    def model(self) -> str:
        return self.config.vision_model

    def configured(self) -> bool:
        return bool(self.config.openai_api_key)

    def _build_prompt(self, question: str, metadata: dict[str, Any]) -> str:
        rules = "\n".join(f"- {rule}" for rule in IMAGE_RULES)
        return f"""Analyze this biomedical image only for observable, research-oriented features.

User question: {question}

Return JSON only with this shape:
{{
  "observations": [
    {{"label": "...", "description": "...", "confidence": null, "location": "...", "caveat": "..."}}
  ]
}}

Rules:
{rules}

Technical image metadata: {json.dumps(metadata, sort_keys=True)}
"""

    def observe(
        self, path: str | Path, question: str
    ) -> tuple[list[ImageObservation], str]:
        # Validate first so an unreadable/oversized image fails before any spend.
        metadata = inspect_image(path, max_pixels=self.config.max_image_pixels)
        if not self.configured():
            raise NotConfiguredError(
                "OPENAI_API_KEY is not configured for multimodal analysis."
            )

        data_url = encode_for_vision(
            path,
            max_dimension=self.config.max_vision_dimension,
            max_pixels=self.config.max_image_pixels,
        )
        payload = {
            "model": self.config.vision_model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": build_system_instruction()},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._build_prompt(question, metadata)},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        }
        body = self._poster(
            f"{self.config.vision_base_url}/chat/completions",
            {
                "Authorization": f"Bearer {self.config.openai_api_key}",
                "Content-Type": "application/json",
            },
            payload,
            self.config.vision_timeout,
        )
        content = self._response_content(body)
        return parse_observations(content), self.config.vision_model

    @staticmethod
    def _response_content(body: dict) -> Any:
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "Vision provider returned an unexpected response shape."
            ) from exc


def observed_terms(observations: Iterable[ImageObservation]) -> list[str]:
    """Concept terms from observations, used to enrich evidence retrieval."""
    terms: list[str] = []
    for obs in observations:
        if obs.label and obs.label != "unspecified":
            terms.append(obs.label)
    return terms
