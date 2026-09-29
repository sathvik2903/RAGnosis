"""Typed exceptions for the multimodal track.

These allow the API layer to distinguish operator/configuration problems (a
missing provider key -> HTTP 503) from genuine upstream failures (HTTP 502) and
from client input problems (HTTP 400), while keeping backward compatibility:
``NotConfiguredError`` subclasses ``RuntimeError`` so existing callers/tests that
catch ``RuntimeError`` keep working.
"""

from __future__ import annotations


class NotConfiguredError(RuntimeError):
    """Raised when a required provider credential/configuration is absent."""
