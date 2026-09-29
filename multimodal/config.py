"""Centralized configuration for the RAGnosis multimodal track.

All environment-driven settings live here so the rest of the package reads
configuration from one documented place instead of scattering ``os.getenv``
calls across modules. Nothing in this module performs network I/O or reads
secrets at import time; values are resolved lazily from the environment when a
``MultimodalConfig`` is constructed. This keeps the package importable (and
unit-testable) without any credentials present.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class MultimodalConfig:
    """Immutable snapshot of the multimodal configuration.

    Construct with :meth:`from_env` to read the current process environment.
    Tests can build an instance directly with explicit values.
    """

    # Vision provider (OpenAI-compatible chat/completions endpoint).
    openai_api_key: str = ""
    vision_base_url: str = "https://api.openai.com/v1"
    vision_model: str = "gpt-4.1-mini"
    vision_timeout: float = 90.0

    # Grounded generation (Cohere Chat API V2).
    cohere_api_key: str = ""
    cohere_model: str = "command-a-plus-05-2026"
    cohere_fallback_model: str = "command-a-reasoning-08-2025"

    # Biomedical retrieval (NCBI PubMed E-utilities).
    pubmed_timeout: float = 15.0
    pubmed_tool: str = "RAGnosis"
    pubmed_email: str = ""
    pubmed_api_key: str = ""

    # Image handling limits.
    max_upload_bytes: int = 12 * 1024 * 1024
    max_image_pixels: int = 40_000_000  # decompression-bomb guard (~40 MP)
    max_vision_dimension: int = 2048  # longest edge sent to the vision model

    # Live health intelligence (public-health surveillance feeds).
    health_timeout: float = 15.0
    health_user_agent: str = "RAGnosis-HealthIntelligence/1.0 (research; contact via repo)"
    health_cache_ttl: float = 900.0  # seconds; 0 disables caching
    health_max_items_per_source: int = 40
    # Freshness thresholds (days). A source newer than "current" is `current`,
    # newer than "recent" is `recent`, otherwise `stale`. Sources that advertise
    # their own cadence should override these at the provider level.
    health_current_days: int = 14
    health_recent_days: int = 60
    # Newline/semicolon/comma separated list of "org|type|scope|url" feed specs.
    # Empty by default so tests and offline use never hit the network implicitly;
    # populate via HEALTH_SOURCE_FEEDS or rely on DEFAULT_HEALTH_FEEDS in code.
    health_source_feeds: str = ""

    @classmethod
    def from_env(cls) -> "MultimodalConfig":
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            vision_base_url=os.getenv(
                "MULTIMODAL_BASE_URL", "https://api.openai.com/v1"
            ).rstrip("/"),
            vision_model=os.getenv("MULTIMODAL_VISION_MODEL", "gpt-4.1-mini"),
            vision_timeout=_get_float("MULTIMODAL_TIMEOUT", 90.0),
            cohere_api_key=os.getenv("COHERE_API_KEY", ""),
            cohere_model=os.getenv(
                "COHERE_MULTIMODAL_RAG_MODEL", "command-a-plus-05-2026"
            ),
            cohere_fallback_model=os.getenv(
                "COHERE_MULTIMODAL_FALLBACK_MODEL", "command-a-reasoning-08-2025"
            ),
            pubmed_timeout=_get_float("PUBMED_TIMEOUT", 15.0),
            pubmed_tool=os.getenv("PUBMED_TOOL", "RAGnosis"),
            pubmed_email=os.getenv("PUBMED_EMAIL", ""),
            pubmed_api_key=os.getenv("PUBMED_API_KEY", ""),
            max_upload_bytes=_get_int(
                "MULTIMODAL_MAX_UPLOAD_BYTES", 12 * 1024 * 1024
            ),
            max_image_pixels=_get_int("MULTIMODAL_MAX_IMAGE_PIXELS", 40_000_000),
            max_vision_dimension=_get_int("MULTIMODAL_MAX_VISION_DIMENSION", 2048),
            health_timeout=_get_float("HEALTH_TIMEOUT", 15.0),
            health_user_agent=os.getenv(
                "HEALTH_USER_AGENT",
                "RAGnosis-HealthIntelligence/1.0 (research; contact via repo)",
            ),
            health_cache_ttl=_get_float("HEALTH_CACHE_TTL", 900.0),
            health_max_items_per_source=_get_int("HEALTH_MAX_ITEMS_PER_SOURCE", 40),
            health_current_days=_get_int("HEALTH_CURRENT_DAYS", 14),
            health_recent_days=_get_int("HEALTH_RECENT_DAYS", 60),
            health_source_feeds=os.getenv("HEALTH_SOURCE_FEEDS", ""),
        )
