#!/usr/bin/env python3
"""Deterministic, offline evaluation entry point (Phase-4 §8).

Usage:
    python evaluation/run_evaluation.py

Runs the full adversarial benchmark against the shipped agent code using
injected fake providers and captured fixtures. Requires no API keys and makes no
live network calls. Exits 0 when no case FAILED, 1 otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
