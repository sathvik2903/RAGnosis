"""End-to-end orchestration tests with fully injected collaborators (no network)."""

from pathlib import Path

import pytest
from PIL import Image

from multimodal.config import MultimodalConfig
from multimodal.retrieval import BiomedicalRetriever
from multimodal.schemas import Evidence, ImageObservation, MultimodalRequest
from multimodal.service import MultimodalRAGService


def _make_image(path: Path) -> Path:
    Image.new("RGB", (32, 32), "gray").save(path)
    return path


class FakeVision:
    def __init__(self, observations=None, model="fake-vision"):
        self.observations = observations or []
        self.model = model
        self.called_with = None

    def configured(self):
        return True

    def observe(self, path, question):
        self.called_with = (path, question)
        return self.observations, self.model


class FakeRetriever:
    def __init__(self, evidence=None, raise_exc=None):
        self.evidence = evidence or []
        self.raise_exc = raise_exc
        self.called = False

    def search(self, question, observations, limit=5, raise_on_error=False):
        self.called = True
        if self.raise_exc:
            raise self.raise_exc
        return self.evidence


class FakeGenerator:
    def __init__(self, text="A grounded, conservative answer.", model="fake-gen"):
        self.text = text
        self.model = model
        self.prompt = None

    def configured(self):
        return True

    def generate(self, prompt):
        self.prompt = prompt
        return self.text, self.model


def _service(vision=None, retriever=None, generator=None):
    return MultimodalRAGService(
        config=MultimodalConfig(),
        vision=vision or FakeVision(),
        retriever=retriever or FakeRetriever(),
        generator=generator or FakeGenerator(),
    )


def test_text_only_pipeline():
    gen = FakeGenerator()
    service = _service(generator=gen)
    result = service.run(MultimodalRequest(question="what causes cough?"))
    assert result.modality == "text"
    assert result.observations == []
    assert result.generation_model == "fake-gen"
    assert "safety note:" in result.answer.lower()


def test_multimodal_pipeline_with_image(tmp_path: Path):
    path = _make_image(tmp_path / "scan.png")
    vision = FakeVision(
        observations=[ImageObservation("opacity", "increased density", 0.3)]
    )
    retriever = FakeRetriever(
        evidence=[
            Evidence("PubMed", "t", "e", metadata={"pmid": "12345678"})
        ]
    )
    service = _service(vision=vision, retriever=retriever)
    result = service.run(
        MultimodalRequest(question="what should be reviewed?", image_path=str(path))
    )
    assert result.modality == "multimodal"
    assert result.observations[0].label == "opacity"
    assert result.retrieval_status == "ok"
    assert result.image_metadata["sha256"]
    assert vision.called_with is not None


def test_retrieval_unavailable_is_non_fatal():
    retriever = FakeRetriever(raise_exc=RuntimeError("pubmed down"))
    service = _service(retriever=retriever)
    result = service.run(MultimodalRequest(question="topic"))
    assert result.retrieval_status == "unavailable"
    assert result.evidence == []
    assert result.answer  # generation still proceeds


def test_retrieval_empty_status():
    retriever = FakeRetriever(evidence=[])
    service = _service(retriever=retriever)
    result = service.run(MultimodalRequest(question="topic"))
    assert result.retrieval_status == "empty"


def test_overconfident_generation_is_withheld_not_returned():
    gen = FakeGenerator(text="This scan confirms cancer.")
    service = _service(generator=gen)
    result = service.run(MultimodalRequest(question="is this cancer?"))
    # The unsafe answer must be enforced (withheld), not merely flagged.
    assert result.safety_action == "withheld"
    assert "confirms cancer" not in result.answer.lower()
    assert any("withheld" in w.lower() for w in result.warnings)


def test_fabricated_citation_is_removed_from_answer():
    gen = FakeGenerator(
        text="This area may warrant review. See PMID 99999999 for details."
    )
    retriever = FakeRetriever(
        evidence=[Evidence("PubMed", "t", "e", metadata={"pmid": "11111111"})]
    )
    service = _service(retriever=retriever, generator=gen)
    result = service.run(MultimodalRequest(question="q"))
    assert result.safety_action == "redacted"
    assert "99999999" not in result.answer
    assert any("removed" in w.lower() for w in result.warnings)


def test_safe_generation_passes_through():
    gen = FakeGenerator(
        text="The findings are nonspecific and should be reviewed by a clinician."
    )
    service = _service(generator=gen)
    result = service.run(MultimodalRequest(question="q"))
    assert result.safety_action == "pass"
    assert "nonspecific" in result.answer


def test_empty_generation_raises():
    gen = FakeGenerator(text="")
    service = _service(generator=gen)
    with pytest.raises(RuntimeError):
        service.run(MultimodalRequest(question="q"))


def test_prompt_includes_retrieval_status():
    gen = FakeGenerator()
    service = _service(generator=gen)
    service.run(MultimodalRequest(question="q"))
    assert "RETRIEVAL STATUS" in gen.prompt


def test_default_collaborators_constructed_from_config():
    # Smoke test: default wiring should not require network at construction.
    service = MultimodalRAGService(MultimodalConfig())
    assert isinstance(service.retriever, BiomedicalRetriever)
    assert service.cohere_model == "command-a-plus-05-2026"
