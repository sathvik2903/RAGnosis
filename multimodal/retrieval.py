from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Iterable
from urllib.parse import quote_plus

import requests

from .config import MultimodalConfig
from .schemas import Evidence, ImageObservation

logger = logging.getLogger("ragnosis.retrieval")


def build_query(question: str, observations: Iterable[ImageObservation]) -> str:
    """Compose a PubMed query from the question plus observed concepts.

    Pure and deterministic so query construction is unit-testable. Observation
    *labels* (short concept names) are prioritized over long descriptions to keep
    the query focused, and the whole thing is length-capped for the E-utilities.
    """
    terms: list[str] = []
    q = (question or "").strip()
    if q:
        terms.append(q)
    for obs in observations:
        label = (obs.label or "").strip()
        if label and label.lower() != "unspecified":
            terms.append(label)
    seen: set[str] = set()
    ordered: list[str] = []
    for term in terms:
        key = term.lower()
        if key not in seen:
            seen.add(key)
            ordered.append(term)
    return " ".join(" ".join(ordered).split())[:900]


def parse_pubmed_xml(xml_text: str) -> list[Evidence]:
    """Parse an efetch PubMed XML payload into Evidence objects.

    Pure function with no network access; deduplicates by PMID and tolerates
    missing titles/abstracts so a partial record still becomes usable evidence.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        logger.warning("Failed to parse PubMed XML payload")
        return []

    evidence: list[Evidence] = []
    seen: set[str] = set()
    for article in root.findall(".//PubmedArticle"):
        pmid = (article.findtext(".//PMID") or "").strip()
        if pmid and pmid in seen:
            continue
        if pmid:
            seen.add(pmid)

        title_node = article.find(".//ArticleTitle")
        title = (
            "".join(title_node.itertext()).strip()
            if title_node is not None
            else ""
        ) or "Untitled PubMed article"

        abstract_parts = [
            "".join(node.itertext())
            for node in article.findall(".//Abstract/AbstractText")
        ]
        excerpt = " ".join(part.strip() for part in abstract_parts).strip()
        if not excerpt:
            excerpt = (
                "Abstract unavailable; retrieve the cited PubMed record for full "
                "text metadata."
            )

        evidence.append(
            Evidence(
                source="PubMed",
                title=title,
                excerpt=excerpt[:1200],
                uri=(
                    f"https://pubmed.ncbi.nlm.nih.gov/{quote_plus(pmid)}/"
                    if pmid
                    else None
                ),
                metadata={"pmid": pmid} if pmid else {},
            )
        )
    return evidence


class RetrievalError(RuntimeError):
    """Raised only when the caller explicitly wants hard failures."""


class BiomedicalRetriever:
    """Evidence retrieval with PubMed as a standards-based external source.

    Retrieval is deliberately separated from generation so every answer can
    expose the evidence that actually influenced it. Because retrieved literature
    is *supporting* evidence (not the answer itself), transient network failures
    degrade gracefully to an empty result rather than aborting the request.
    """

    def __init__(
        self,
        config: MultimodalConfig | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.config = config or MultimodalConfig.from_env()
        self.base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self._session = session or requests.Session()

    @property
    def timeout(self) -> float:
        return self.config.pubmed_timeout

    def _common_params(self) -> dict[str, str]:
        # NCBI etiquette: identify the tool and (optionally) supply an API key,
        # which raises the request rate limit and reduces throttling.
        params: dict[str, str] = {"tool": self.config.pubmed_tool}
        if self.config.pubmed_email:
            params["email"] = self.config.pubmed_email
        if self.config.pubmed_api_key:
            params["api_key"] = self.config.pubmed_api_key
        return params

    def search(
        self,
        question: str,
        observations: list[ImageObservation],
        limit: int = 5,
        raise_on_error: bool = False,
    ) -> list[Evidence]:
        query = build_query(question, observations)
        if not query:
            return []

        try:
            ids = self._esearch(query, limit)
            if not ids:
                return []
            xml_text = self._efetch(ids)
        except (requests.RequestException, ValueError) as exc:
            logger.warning("PubMed retrieval failed: %s", exc)
            if raise_on_error:
                raise RetrievalError(str(exc)) from exc
            return []

        return parse_pubmed_xml(xml_text)

    def _esearch(self, query: str, limit: int) -> list[str]:
        params = self._common_params()
        params.update(
            {
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": str(max(1, min(limit, 10))),
                "sort": "relevance",
            }
        )
        resp = self._session.get(
            f"{self.base}/esearch.fcgi", params=params, timeout=self.timeout
        )
        resp.raise_for_status()
        ids = resp.json().get("esearchresult", {}).get("idlist", [])
        return [str(i) for i in ids if str(i).strip()]

    def _efetch(self, ids: list[str]) -> str:
        params = self._common_params()
        params.update({"db": "pubmed", "id": ",".join(ids), "retmode": "xml"})
        resp = self._session.get(
            f"{self.base}/efetch.fcgi", params=params, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.text
