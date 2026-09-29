"""PubMed query-building, XML parsing, and network-degradation tests."""

import requests

from multimodal.config import MultimodalConfig
from multimodal.retrieval import (
    BiomedicalRetriever,
    build_query,
    parse_pubmed_xml,
)
from multimodal.schemas import ImageObservation

SAMPLE_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345678</PMID>
      <Article>
        <ArticleTitle>Imaging features of pulmonary nodules</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">Nodules are common.</AbstractText>
          <AbstractText Label="RESULTS">CT helps characterization.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345678</PMID>
      <Article><ArticleTitle>Duplicate should be dropped</ArticleTitle></Article>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>87654321</PMID>
      <Article><ArticleTitle>No abstract article</ArticleTitle></Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""


def test_build_query_combines_and_dedupes():
    obs = [ImageObservation("nodule", "d"), ImageObservation("nodule", "d2")]
    query = build_query("lung imaging", obs)
    assert "lung imaging" in query
    assert query.lower().count("nodule") == 1


def test_build_query_skips_unspecified():
    obs = [ImageObservation("unspecified", "d")]
    assert build_query("chest", obs) == "chest"


def test_build_query_empty():
    assert build_query("", []) == ""


def test_parse_pubmed_xml_basic():
    evidence = parse_pubmed_xml(SAMPLE_XML)
    # Duplicate PMID collapsed -> 2 unique records.
    assert len(evidence) == 2
    first = evidence[0]
    assert first.metadata["pmid"] == "12345678"
    assert "Nodules are common" in first.excerpt
    assert first.uri == "https://pubmed.ncbi.nlm.nih.gov/12345678/"


def test_parse_pubmed_xml_missing_abstract():
    evidence = parse_pubmed_xml(SAMPLE_XML)
    no_abstract = [e for e in evidence if e.metadata["pmid"] == "87654321"][0]
    assert "Abstract unavailable" in no_abstract.excerpt


def test_parse_pubmed_xml_malformed_returns_empty():
    assert parse_pubmed_xml("<not-valid") == []
    assert parse_pubmed_xml("") == []


class _FakeResponse:
    def __init__(self, *, json_data=None, text="", raise_exc=None):
        self._json = json_data
        self.text = text
        self._raise = raise_exc

    def raise_for_status(self):
        if self._raise:
            raise self._raise

    def json(self):
        return self._json


class _FakeSession:
    def __init__(self, esearch_ids=None, efetch_xml="", fail=False):
        self.esearch_ids = esearch_ids or []
        self.efetch_xml = efetch_xml
        self.fail = fail
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        if self.fail:
            raise requests.ConnectionError("network down")
        if "esearch" in url:
            return _FakeResponse(
                json_data={"esearchresult": {"idlist": self.esearch_ids}}
            )
        return _FakeResponse(text=self.efetch_xml)


def test_search_returns_evidence():
    session = _FakeSession(esearch_ids=["12345678"], efetch_xml=SAMPLE_XML)
    retriever = BiomedicalRetriever(MultimodalConfig(), session=session)
    evidence = retriever.search("lung nodule", [])
    assert evidence
    assert evidence[0].metadata["pmid"] == "12345678"


def test_search_no_ids_returns_empty():
    session = _FakeSession(esearch_ids=[])
    retriever = BiomedicalRetriever(MultimodalConfig(), session=session)
    assert retriever.search("nothing", []) == []


def test_search_degrades_on_network_error():
    session = _FakeSession(fail=True)
    retriever = BiomedicalRetriever(MultimodalConfig(), session=session)
    # Default behaviour: swallow the error and return no evidence.
    assert retriever.search("lung", []) == []


def test_search_raises_when_requested():
    session = _FakeSession(fail=True)
    retriever = BiomedicalRetriever(MultimodalConfig(), session=session)
    try:
        retriever.search("lung", [], raise_on_error=True)
    except RuntimeError:
        return
    raise AssertionError("expected RetrievalError")


class _BadJsonResponse:
    text = "<html>not json</html>"

    def raise_for_status(self):
        return None

    def json(self):
        raise ValueError("Expecting value: line 1 column 1 (char 0)")


def test_search_degrades_on_malformed_json():
    class _S:
        def get(self, url, params=None, timeout=None):
            return _BadJsonResponse()

    retriever = BiomedicalRetriever(MultimodalConfig(), session=_S())
    # A non-JSON esearch body raises ValueError, which must degrade to [].
    assert retriever.search("lung", []) == []


def test_search_http_error_degrades():
    class _S:
        def get(self, url, params=None, timeout=None):
            resp = _FakeResponse(
                raise_exc=requests.HTTPError("429 Too Many Requests")
            )
            return resp

    retriever = BiomedicalRetriever(MultimodalConfig(), session=_S())
    assert retriever.search("lung", []) == []


def test_common_params_include_etiquette():
    config = MultimodalConfig(pubmed_email="me@example.com", pubmed_api_key="abc")
    session = _FakeSession(esearch_ids=["1"], efetch_xml=SAMPLE_XML)
    retriever = BiomedicalRetriever(config, session=session)
    retriever.search("q", [])
    _, params = session.calls[0]
    assert params["tool"] == "RAGnosis"
    assert params["email"] == "me@example.com"
    assert params["api_key"] == "abc"
