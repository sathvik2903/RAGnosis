"""Deterministic routing tests (§13)."""

from multimodal.routing import classify


def test_static_question_is_literature_only():
    d = classify("what causes dengue?", has_image=False, has_location=False)
    assert d.capabilities == ["literature"]
    assert d.live_health_intent is False


def test_current_outbreak_question_invokes_health():
    d = classify(
        "is there a dengue outbreak right now?", has_image=False, has_location=False
    )
    assert "health_intelligence" in d.capabilities
    assert d.live_health_intent is True
    # Outbreak questions are inherently geographic even without a place.
    assert d.geographic_intent is True


def test_contagion_question_invokes_health():
    d = classify(
        "is measles contagious currently?", has_image=False, has_location=False
    )
    assert d.live_health_intent is True
    assert "health_intelligence" in d.capabilities


def test_image_adds_vision():
    d = classify(
        "what should be reviewed in this image?", has_image=True, has_location=False
    )
    assert "vision" in d.capabilities
    assert "literature" in d.capabilities


def test_combined_vision_literature_health():
    d = classify(
        "describe this rash and tell me about current outbreaks in my area",
        has_image=True,
        has_location=True,
    )
    assert set(d.capabilities) >= {"vision", "literature", "health_intelligence"}
    assert d.geographic_intent is True


def test_explicit_location_with_health_question_routes_health():
    d = classify(
        "any diseases I should watch for", has_image=False, has_location=True
    )
    assert "health_intelligence" in d.capabilities


def test_static_with_location_stays_literature():
    # A definitional question shouldn't be forced into a live lookup just because
    # a location was provided.
    d = classify("what is dengue?", has_image=False, has_location=True)
    assert d.live_health_intent is False
    assert "health_intelligence" not in d.capabilities


def test_geographic_cue_detected():
    d = classify(
        "current health alerts in my region", has_image=False, has_location=False
    )
    assert d.geographic_intent is True
    assert d.live_health_intent is True


def test_route_label_lists_capabilities():
    d = classify("outbreak now in India", has_image=False, has_location=True)
    assert "health_intelligence" in d.route
    assert "literature" in d.route
