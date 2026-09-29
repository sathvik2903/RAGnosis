from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from .config import MultimodalConfig
from .errors import NotConfiguredError
from .image import inspect_image
from .retrieval import BiomedicalRetriever
from .safety import build_system_instruction, validate_response
from .schemas import (
    Evidence,
    ImageObservation,
    MultimodalRequest,
    MultimodalResponse,
    RetrievalStatus,
)
from .vision import VisionProvider

logger = logging.getLogger("ragnosis.multimodal")

BASE_LIMITATIONS = (
    "Image observations are not diagnoses or confirmed clinical findings.",
    "Retrieved literature is supporting evidence, not patient-specific medical advice.",
    "Clinical decisions require qualified professional review of the complete patient record.",
)


class TextGenerator(Protocol):
    """Minimal interface the service needs from a text generation backend."""

    def generate(self, prompt: str) -> tuple[str, str]:  # (text, model_used)
        ...

    def configured(self) -> bool:
        ...


class CohereGenerator:
    """Cohere-backed generator with primary/fallback model handling.

    Uses the Cohere Chat API **V2** (``cohere.ClientV2`` + ``messages=[...]``);
    the current command models are only served on ``/v2/chat``. Imports the
    Cohere SDK lazily so the package remains importable (and unit testable via an
    injected generator) without the dependency present.
    """

    def __init__(self, config: MultimodalConfig | None = None) -> None:
        self.config = config or MultimodalConfig.from_env()
        self._client = None

    def configured(self) -> bool:
        return bool(self.config.cohere_api_key)

    def _get_client(self):
        if not self.configured():
            raise NotConfiguredError(
                "COHERE_API_KEY is not configured for multimodal RAG generation."
            )
        if self._client is None:
            import cohere  # local import keeps the dependency optional at import time

            self._client = cohere.ClientV2(self.config.cohere_api_key)
        return self._client

    @staticmethod
    def _extract_text(response) -> str:
        """Read text from a Cohere V2 chat response (``message.content[0].text``).

        Defensive against missing/empty content so an empty response is handled
        by the caller rather than raising an ``AttributeError``/``IndexError``.
        """
        message = getattr(response, "message", None)
        content = getattr(message, "content", None) or []
        parts: list[str] = []
        for block in content:
            text = getattr(block, "text", None)
            if text is None and isinstance(block, dict):
                text = block.get("text")
            if text:
                parts.append(text)
        return "".join(parts).strip()

    def _chat(self, prompt: str, model: str) -> str:
        response = self._get_client().chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return self._extract_text(response)

    def generate(self, prompt: str) -> tuple[str, str]:
        client_model = self.config.cohere_model
        try:
            text = self._chat(prompt, client_model)
            if text:
                return text, client_model
            raise RuntimeError("The generation provider returned an empty response.")
        except Exception as exc:  # noqa: BLE001 - decide fallback vs. re-raise below
            message = str(exc).lower()
            model_error = any(
                token in message
                for token in ("model", "not found", "decommissioned", "unknown")
            )
            fallback = self.config.cohere_fallback_model
            if model_error and fallback and fallback != client_model:
                logger.warning("Primary generation model failed; trying fallback")
                text = self._chat(prompt, fallback)
                if text:
                    return text, fallback
            raise


class MultimodalRAGService:
    """Orchestrates validation -> vision -> retrieval -> grounded generation.

    Each collaborator is injectable so the full pipeline can be exercised
    offline. In production, defaults read configuration from the environment.
    """

    def __init__(
        self,
        config: MultimodalConfig | None = None,
        vision: VisionProvider | None = None,
        retriever: BiomedicalRetriever | None = None,
        generator: TextGenerator | None = None,
    ) -> None:
        self.config = config or MultimodalConfig.from_env()
        self.vision = vision or VisionProvider(self.config)
        self.retriever = retriever or BiomedicalRetriever(self.config)
        self.generator = generator or CohereGenerator(self.config)

    # Kept for backward compatibility with existing callers/health checks.
    @property
    def cohere_key(self) -> str:
        return self.config.cohere_api_key

    @property
    def cohere_model(self) -> str:
        return self.config.cohere_model

    def run(self, request: MultimodalRequest) -> MultimodalResponse:
        warnings: list[str] = []
        observations: list[ImageObservation] = []
        vision_model: str | None = None
        image_metadata: dict[str, Any] = {}

        if request.image_path:
            image_metadata = inspect_image(
                request.image_path, max_pixels=self.config.max_image_pixels
            )
            observations, vision_model = self.vision.observe(
                request.image_path, request.question
            )

        evidence, retrieval_status = self._retrieve(request.question, observations)

        prompt = self._build_prompt(request, observations, evidence, retrieval_status)
        answer, generation_model = self.generator.generate(prompt)
        if not answer:
            raise RuntimeError("The generation provider returned an empty response.")

        # Deterministic post-generation guard: verify the model did not overstep
        # the safety contract or cite evidence it was never given. When it did,
        # the unsafe answer is withheld/redacted here rather than returned.
        validation = validate_response(answer, evidence, observations)
        warnings.extend(validation.warnings)
        if validation.enforced:
            logger.warning(
                "Safety enforcement applied (%s): %s",
                validation.action.value,
                "; ".join(validation.warnings),
            )

        return MultimodalResponse(
            answer=validation.text,
            modality=request.modality,
            observations=observations,
            evidence=evidence,
            limitations=list(BASE_LIMITATIONS),
            model=vision_model,
            generation_model=generation_model,
            retrieval_status=retrieval_status,
            warnings=warnings,
            image_metadata=image_metadata,
            safety_action=validation.action.value,
        )

    def _retrieve(
        self, question: str, observations: list[ImageObservation]
    ) -> tuple[list[Evidence], RetrievalStatus]:
        query_present = bool((question or "").strip() or observations)
        if not query_present:
            return [], "skipped"
        try:
            evidence = self.retriever.search(
                question, observations, raise_on_error=True
            )
        except Exception as exc:  # noqa: BLE001 - retrieval is non-fatal by design
            logger.warning("Evidence retrieval unavailable: %s", exc)
            return [], "unavailable"
        return (evidence, "ok") if evidence else ([], "empty")

    def _build_prompt(
        self,
        request: MultimodalRequest,
        observations: list[ImageObservation],
        evidence: list[Evidence],
        retrieval_status: RetrievalStatus,
    ) -> str:
        observation_text = json.dumps([o.to_dict() for o in observations], indent=2)
        evidence_text = json.dumps([e.to_dict() for e in evidence], indent=2)
        conversation = json.dumps(request.conversation[-8:], indent=2)
        retrieval_note = {
            "ok": "Literature evidence was retrieved and is provided below.",
            "empty": "No relevant literature was found; do not invent citations.",
            "unavailable": "Literature retrieval was unavailable; do not invent citations.",
            "skipped": "No literature retrieval was performed.",
        }[retrieval_status]
        return f"""{build_system_instruction()}

USER QUESTION:
{request.question}

RECENT CONVERSATION:
{conversation}

MODEL IMAGE OBSERVATIONS:
{observation_text}

RETRIEVED BIOMEDICAL EVIDENCE:
{evidence_text}

RETRIEVAL STATUS: {retrieval_status} - {retrieval_note}

Write a concise, evidence-grounded response. Explicitly separate image
observations from conclusions supported by literature. Only cite PubMed records
that appear in the retrieved evidence above; never invent a PMID or source. If
the evidence does not support a conclusion, say so. Never turn a visual
observation into a definitive diagnosis. Include PubMed links when they are
present in the evidence metadata.
"""
