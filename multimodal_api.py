from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from multimodal.agent import AgentService
from multimodal.config import MultimodalConfig
from multimodal.errors import NotConfiguredError
from multimodal.image import ImageValidationError, SUPPORTED_IMAGE_TYPES
from multimodal.schemas import MultimodalRequest
from multimodal.service import MultimodalRAGService

logger = logging.getLogger("ragnosis.multimodal_api")

ROOT_DIR = Path(__file__).resolve().parent
CONFIG = MultimodalConfig.from_env()
app = Flask(__name__)
# Reject oversized uploads at the WSGI layer before buffering the whole body.
app.config["MAX_CONTENT_LENGTH"] = CONFIG.max_upload_bytes
service = MultimodalRAGService(CONFIG)
# The agent reuses the same multimodal service and adds live health intelligence
# as one more tool (compose, don't fork). No network happens at construction.
agent = AgentService(CONFIG, multimodal=service)


@app.get("/")
def index():
    """Serve a self-contained demo UI so a judge can use the API in a browser."""
    return send_from_directory(ROOT_DIR / "multimodal", "demo.html")


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "service": "ragnosis-multimodal",
            "vision_configured": service.vision.configured(),
            "generation_configured": service.generator.configured(),
            "health_intelligence_configured": agent.health.configured(),
        }
    )


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": "image exceeds upload limit"}), 413


@app.post("/analyze")
def analyze():
    question = (request.form.get("question") or "").strip()
    image = request.files.get("image")
    if not question:
        return jsonify({"error": "question is required"}), 400
    if image is None:
        return jsonify({"error": "image is required"}), 400

    suffix = Path(image.filename or "image.png").suffix.lower() or ".png"
    if suffix not in SUPPORTED_IMAGE_TYPES:
        return jsonify(
            {
                "error": (
                    f"unsupported image type '{suffix}'. supported: "
                    f"{', '.join(sorted(SUPPORTED_IMAGE_TYPES))}"
                )
            }
        ), 400

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            image.save(handle)
            temp_path = handle.name
        result = service.run(MultimodalRequest(question=question, image_path=temp_path))
        return jsonify(result.to_dict())
    except ImageValidationError as exc:
        # Client-side problem with the uploaded image -> 400, not 502.
        return jsonify({"error": str(exc)}), 400
    except NotConfiguredError as exc:
        # Missing provider configuration is a deployment/operator issue; the
        # message is safe (it names an env var, not a secret) and actionable.
        logger.error("multimodal analysis unavailable: %s", exc)
        return jsonify({"error": str(exc)}), 503
    except Exception:  # noqa: BLE001 - upstream/provider failure
        # Log full detail server-side; return a generic message so provider
        # internals / stack details are never leaked to the client.
        logger.exception("multimodal analysis failed")
        return jsonify(
            {"error": "Upstream analysis failed. Please retry later."}
        ), 502
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


@app.post("/agent")
def agent_analyze():
    """Composed agent: question + optional image + optional explicit location.

    Unlike /analyze, the image is optional here (text/live-health questions need
    no image) and an explicit ``location`` field scopes any regional public
    health lookup. Location is NEVER inferred server-side.
    """
    question = (request.form.get("question") or "").strip()
    location_text = (request.form.get("location") or "").strip() or None
    image = request.files.get("image")
    if not question:
        return jsonify({"error": "question is required"}), 400

    temp_path = None
    try:
        if image is not None and image.filename:
            suffix = Path(image.filename).suffix.lower() or ".png"
            if suffix not in SUPPORTED_IMAGE_TYPES:
                return jsonify(
                    {
                        "error": (
                            f"unsupported image type '{suffix}'. supported: "
                            f"{', '.join(sorted(SUPPORTED_IMAGE_TYPES))}"
                        )
                    }
                ), 400
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                image.save(handle)
                temp_path = handle.name

        result = agent.run(
            MultimodalRequest(question=question, image_path=temp_path),
            location_text=location_text,
        )
        return jsonify(result.to_dict())
    except ImageValidationError as exc:
        return jsonify({"error": str(exc)}), 400
    except NotConfiguredError as exc:
        logger.error("agent analysis unavailable: %s", exc)
        return jsonify({"error": str(exc)}), 503
    except Exception:  # noqa: BLE001 - upstream/provider failure
        logger.exception("agent analysis failed")
        return jsonify(
            {"error": "Upstream analysis failed. Please retry later."}
        ), 502
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8001")), debug=False)
