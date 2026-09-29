import json
import logging
import os
import re
import threading
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from neo4j import GraphDatabase
import cohere

# Reuse the already-validated multimodal + agent HTTP layer instead of forking
# it. Importing the module builds its CONFIG/service/agent once (no network at
# construction) and exposes the /agent and /analyze view functions, the 413
# handler, and the upload limit that this canonical entrypoint mounts below.
# multimodal_api keeps its own app/service/agent so its contract tests and local
# development server continue to work unchanged.
import multimodal_api

logging.getLogger("neo4j").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ragnosis")

ROOT_DIR = Path(__file__).resolve().parent

NEO4J_URI = os.environ.get("NEO4J_URI", "")
NEO4J_USER = os.environ.get("NEO4J_USER", "")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "")


def neo4j_settings():
    return {
        "uri": os.environ.get("NEO4J_URI", ""),
        "user": os.environ.get("NEO4J_USER", ""),
        "password": os.environ.get("NEO4J_PASSWORD", ""),
        "database": os.environ.get("NEO4J_DATABASE", ""),
    }


def cohere_api_key():
    return os.environ.get("COHERE_API_KEY", "")


def _redact(text, secret):
    text = str(text)
    if secret and secret in text:
        return text.replace(secret, "[redacted]")
    return text


SAFE_GENERATION_MESSAGE = (
    "I'm having trouble processing your request right now. "
    "Please try again in a moment."
)
SAFE_RETRIEVAL_MESSAGE = (
    "I'm having trouble accessing the medical knowledge graph right now. "
    "Please try again in a moment."
)


class GenerationUnavailableError(RuntimeError):
    """Controlled failure that is safe to surface through the /chat API."""

    def __init__(self, code="generation_unavailable"):
        self.code = code
        super().__init__(code)


def _status_code(exc):
    value = getattr(exc, "status_code", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_rate_limited(exc):
    return _status_code(exc) == 429 or type(exc).__name__ == "TooManyRequestsError"


def _is_model_unavailable(exc):
    status = _status_code(exc)
    name = type(exc).__name__
    message = str(exc).lower()
    return (
        status == 404
        or name in {"NotFoundError", "ModelNotFoundError"}
        or "model not found" in message
        or "decommissioned" in message
    )


PRIMARY_COHERE_MODEL = "command-a-03-2025"
FALLBACK_COHERE_MODEL = "command-r7b-12-2024"

MEDICAL_DISCLAIMER = (
    "RAGnosis is an informational biomedical assistant. It does not provide "
    "professional medical diagnosis or treatment and is not a substitute for "
    "qualified clinical care. Seek in-person medical help for personal health decisions."
)

SEARCH_PROPS = ["name", "text", "description", "disease", "symptom", "title", "canonical_name"]

DIAGNOSIS_MEDICAL_PHRASES = sorted(
    [
        "shortness of breath",
        "difficulty breathing",
        "trouble breathing",
        "labored breathing",
        "loss of taste",
        "loss of smell",
        "loss of appetite",
        "loss of consciousness",
        "loss of balance",
        "loss of vision",
        "high blood pressure",
        "low blood pressure",
        "high blood sugar",
        "low blood sugar",
        "rapid heart rate",
        "irregular heart rate",
        "irregular heartbeat",
        "heart palpitations",
        "racing heart",
        "chest pain",
        "chest tightness",
        "chest pressure",
        "chest discomfort",
        "sore throat",
        "scratchy throat",
        "swollen throat",
        "throat irritation",
        "high fever",
        "low grade fever",
        "mild fever",
        "spiking fever",
        "abdominal pain",
        "stomach pain",
        "stomach ache",
        "belly pain",
        "pelvic pain",
        "muscle pain",
        "muscle ache",
        "muscle aches",
        "joint pain",
        "joint swelling",
        "joint stiffness",
        "body aches",
        "body ache",
        "runny nose",
        "stuffy nose",
        "nasal congestion",
        "sinus congestion",
        "sinus pressure",
        "sinus pain",
        "dry cough",
        "productive cough",
        "chronic cough",
        "barking cough",
        "whooping cough",
        "coughing up blood",
        "blood in sputum",
        "blood in stool",
        "blood in urine",
        "night sweats",
        "cold sweats",
        "cold chills",
        "chills and fever",
        "weight loss",
        "weight gain",
        "blurred vision",
        "blurry vision",
        "double vision",
        "skin rash",
        "itchy rash",
        "itchy skin",
        "difficulty swallowing",
        "painful swallowing",
        "swollen lymph nodes",
        "swollen glands",
        "swollen tonsils",
        "swollen ankles",
        "swollen feet",
        "swollen legs",
        "acid reflux",
        "heart burn",
        "heartburn",
        "ear pain",
        "ear ache",
        "earache",
        "neck pain",
        "neck stiffness",
        "stiff neck",
        "back pain",
        "lower back pain",
        "upper back pain",
        "head ache",
        "hoarse voice",
        "dry mouth",
        "frequent urination",
        "painful urination",
        "burning urination",
        "mood swings",
        "memory loss",
        "brain fog",
        "pale skin",
        "yellow skin",
        "yellow eyes",
        "hair loss",
        "numbness and tingling",
        "pins and needles",
        "light headedness",
        "light headed",
        "upper respiratory tract infection",
        "upper respiratory infection",
        "urinary tract infection",
        "strep throat",
        "sinus infection",
        "ear infection",
        "food poisoning",
        "allergic reaction",
        "asthma attack",
        "panic attack",
        "heart attack",
    ],
    key=len,
    reverse=True,
)

DIAGNOSIS_STOP_WORDS = {
    # Pronouns & determiners
    "i", "me", "my", "myself", "we", "us", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "whose", "this", "that", "these", "those",
    "a", "an", "the", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",

    # Prepositions & Conjunctions
    "about", "above", "across", "after", "afterwards", "again", "against",
    "along", "already", "also", "although", "always", "among", "amongst",
    "and", "around", "as", "at", "because", "before", "behind", "below",
    "beside", "besides", "between", "beyond", "but", "by", "down", "during",
    "except", "for", "from", "in", "inside", "into", "near", "of", "off",
    "on", "onto", "or", "out", "outside", "over", "since", "so", "than",
    "then", "there", "therefore", "through", "throughout", "to", "toward",
    "towards", "under", "until", "up", "upon", "with", "within", "without",

    # Verbs, auxiliaries & modals
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing", "done",
    "can", "could", "shall", "should", "will", "would", "may", "might", "must",

    # Contractions fragments
    "ve", "re", "ll", "don", "didn", "doesn", "won", "wasn", "weren", "isn", "aren", "hasn", "haven", "hadn",

    # Clinical & conversational framing
    "patient", "patients", "person", "people", "man", "woman", "child", "children",
    "report", "reports", "reported", "reporting",
    "complain", "complains", "complained", "complaining", "complaint", "complaints",
    "feel", "feeling", "feels", "felt",
    "seem", "seems", "seemed", "seeming",
    "look", "looks", "looked", "looking",
    "get", "gets", "got", "getting", "gotten",
    "go", "goes", "went", "gone", "going",
    "come", "comes", "came", "coming",
    "start", "starts", "started", "starting",
    "experience", "experiences", "experienced", "experiencing",
    "suffer", "suffers", "suffered", "suffering",
    "notice", "notices", "noticed", "noticing",
    "think", "thinks", "thought", "thinking",
    "know", "knows", "knew", "knowing",
    "tell", "tells", "told", "telling",
    "ask", "asks", "asked", "asking",
    "say", "says", "said", "saying",
    "want", "wants", "wanted", "wanting",
    "need", "needs", "needed", "needing",
    "try", "tries", "tried", "trying",
    "take", "takes", "took", "taken", "taking",
    "give", "gives", "gave", "given", "giving",
    "make", "makes", "made", "making",
    "find", "finds", "found", "finding",
    "see", "sees", "saw", "seen", "seeing",
    "help", "please", "hello", "hi", "hey", "doc", "doctor",
    "thanks", "thank", "thankyou",
    "yes", "yeah", "yep", "sure", "ok", "okay",
    "maybe", "perhaps", "probably", "possibly",
    "really", "very", "quite", "extremely", "pretty", "fairly",
    "just", "like", "well", "now", "still", "even",
    "bad", "worse", "worst", "better", "good", "fine", "terrible", "awful", "horrible",
    "severe", "severity", "mild", "moderate", "slight", "slightly",
    "little", "bit", "lot", "lots", "much", "many", "less", "least",
    "day", "days", "week", "weeks", "month", "months", "year", "years",
    "hour", "hours", "minute", "minutes", "second", "seconds", "time", "times",
    "today", "yesterday", "tomorrow", "tonight", "morning", "afternoon", "evening", "night", "nights",
    "daily", "weekly", "monthly",
    "ago", "past", "last", "lately", "recently", "recent", "current", "currently",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
    "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand",
    "first", "second", "third", "fourth", "fifth", "couple", "several", "half", "double",
    "how", "why", "when", "where",
    "something", "anything", "nothing", "everything",
    "someone", "anyone", "everyone",
    "somewhere", "anywhere",
    "problem", "issue", "issues", "symptom", "symptoms", "condition", "conditions",
    "illness", "disease", "diseases",
    "trouble", "worried", "worry", "wondering",
}


def extract_diagnosis_terms(text: str, max_terms: int = 6) -> list[str]:
    """Deterministic, stdlib-only symptom/condition term extractor.

    Extracts multi-word medical phrases first (longest match first) and then
    retains meaningful single medical/symptom tokens while filtering out
    conversational, temporal, numeric, and grammatical filler words.
    """
    if not text or not isinstance(text, str):
        return []

    cleaned = re.sub(r"[^a-z]+", " ", text.lower())
    padded = f" {cleaned} "

    matched_spans = []  # list of (start_idx, end_idx, phrase)

    for phrase in DIAGNOSIS_MEDICAL_PHRASES:
        pattern = r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])"
        for match in re.finditer(pattern, padded):
            start, end = match.start(), match.end()
            if not any(s < end and start < e for s, e, _ in matched_spans):
                matched_spans.append((start, end, phrase))

    chars = list(padded)
    for start, end, _ in matched_spans:
        for i in range(start, end):
            chars[i] = " "
    unmatched = "".join(chars)

    single_word_matches = []
    for match in re.finditer(r"[a-z]{2,}", unmatched):
        word = match.group()
        if word not in DIAGNOSIS_STOP_WORDS:
            single_word_matches.append((match.start(), match.end(), word))

    all_matches = sorted(matched_spans + single_word_matches, key=lambda x: x[0])

    seen = set()
    deduped = []
    for _, _, term in all_matches:
        term = term.strip()
        if term and term not in seen:
            seen.add(term)
            deduped.append(term)
            if len(deduped) >= max_terms:
                break

    return deduped


def _jsonable(value, limit=800):
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "..."
    if isinstance(value, list):
        if value and isinstance(value[0], (int, float)) and len(value) > 32:
            return None
        return [_jsonable(v, limit=limit) for v in value[:12]]
    if isinstance(value, dict):
        return {k: _jsonable(v, limit=limit) for k, v in value.items()}
    return str(value)


def _node_properties(node):
    props = {}
    raw = getattr(node, "_properties", None)
    if raw is None:
        try:
            raw = dict(node)
        except Exception:
            return props
    for key, value in raw.items():
        if key.lower() in {"embedding", "embeddings", "vector"}:
            continue
        converted = _jsonable(value)
        if converted is None:
            continue
        props[key] = converted
    return props


class Neo4jConnector:
    def __init__(self, uri, user, password, database=""):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None
        self.last_error = ""

    def configured(self):
        return bool(self.uri and self.user and self.password)

    def _session(self):
        if self.database:
            return self.driver.session(database=self.database)
        return self.driver.session()

    def connect(self):
        if not self.configured():
            self.last_error = "Neo4j environment variables are not set."
            return False
        if self.driver is not None:
            return True
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                connection_timeout=10,
            )
            self.driver.verify_connectivity()
            self.last_error = ""
            logger.info("Connected to Neo4j Aura")
            return True
        except Exception as exc:
            self.last_error = _redact(exc, self.password)
            logger.error("Neo4j connection failed")
            if self.driver is not None:
                try:
                    self.driver.close()
                except Exception:
                    pass
                self.driver = None
            return False

    def close(self):
        if self.driver is not None:
            self.driver.close()
            self.driver = None

    def ping(self):
        if not self.connect():
            return False
        try:
            with self._session() as session:
                session.run("RETURN 1 AS ok").single()
            self.last_error = ""
            return True
        except Exception as exc:
            self.last_error = _redact(exc, self.password)
            return False

    def search_entities(self, query_text):
        if not self.connect():
            raise RuntimeError(self.last_error or "Neo4j is not connected.")

        query_text = (query_text or "").strip()
        if not query_text:
            return "No relevant entities found."

        terms = extract_diagnosis_terms(query_text)
        if not terms:
            terms = [query_text[:300]]

        query_parts = [
            f"(n.{prop} IS NOT NULL AND toLower(toString(n.{prop})) CONTAINS toLower($term))"
            for prop in SEARCH_PROPS
        ]
        where_clause = " OR ".join(query_parts)
        cypher_query = f"""
            MATCH (n)
            WHERE {where_clause}
            RETURN n
            LIMIT 5
        """

        matched_entities = {}
        order_counter = 0

        with self._session() as session:
            for term in terms:
                result = session.run(cypher_query, {"term": term[:300]})
                records = result.values() if hasattr(result, "values") else list(result)
                for row in records:
                    if not row:
                        continue
                    if isinstance(row, (list, tuple)):
                        node = row[0]
                    elif hasattr(row, "__getitem__") and "n" in row:
                        node = row["n"]
                    elif hasattr(row, "__getitem__"):
                        try:
                            node = row[0]
                        except Exception:
                            node = row
                    else:
                        node = row
                    props = _node_properties(node)
                    if not props:
                        continue
                    key = json.dumps(props, sort_keys=True)
                    if key not in matched_entities:
                        matched_entities[key] = {
                            "props": props,
                            "terms": {term},
                            "order": order_counter,
                        }
                        order_counter += 1
                    else:
                        matched_entities[key]["terms"].add(term)

        if not matched_entities:
            return "No relevant entities found."

        ranked = sorted(
            matched_entities.values(),
            key=lambda item: (-len(item["terms"]), item["order"]),
        )

        top_entities = ranked[:5]
        context = "\n".join(
            [
                json.dumps(item["props"], indent=2)
                for item in top_entities
            ]
        )
        return context if context else "No relevant entities found."


class BiomedicalRAG:
    def __init__(self, api_key, model=PRIMARY_COHERE_MODEL):
        self.api_key = api_key or ""
        self.model = model
        self.client = None
        self.last_error = ""

    def configured(self):
        return bool(self.api_key)

    def _client(self):
        if not self.configured():
            raise RuntimeError("COHERE_API_KEY is not set.")
        if self.client is None:
            self.client = cohere.Client(self.api_key)
        return self.client

    def _prompt(self, conversation, context):
        convo_text = "\n".join(
            [
                f"Patient: {c['patient']}\nDoctor: {c.get('doctor', '')}"
                if "doctor" in c
                else f"Patient: {c['patient']}"
                for c in conversation
            ]
        )
        return f"""
You are a kind, logical, biomedical doctor chatbot.

SAFETY:
- You are an informational assistant, not a licensed clinician.
- Do not claim to replace professional diagnosis or treatment.
- Remind the user to seek qualified medical care for personal health decisions.

PATIENT CONVERSATION HISTORY:
{convo_text}

KNOWLEDGE GRAPH CONTEXT:
{context}

TASK:
- Ask relevant medical follow-up questions.
- If enough info, provide the most likely diagnosis and advice.
- Be concise, empathetic, and medically accurate.
- Ground answers in the knowledge graph context when it is available.
- If the knowledge graph context is missing or insufficient, say so.
- End the chat when confident in your diagnosis.

YOUR RESPONSE:
"""

    def _chat(self, prompt, model):
        response = self._client().chat(
            model=model,
            message=prompt,
            temperature=0.4,
        )
        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError("Cohere returned an empty response.")
        return text.strip()

    def answer(self, conversation, context):
        if not self.configured():
            self.last_error = "not_configured"
            logger.warning("Cohere generation unavailable: key not configured")
            raise GenerationUnavailableError("generation_not_configured")

        prompt = self._prompt(conversation, context)
        try:
            return self._chat(prompt, self.model)
        except Exception as exc:
            if _is_rate_limited(exc):
                self.last_error = "rate_limited"
                logger.warning(
                    "Cohere generation rate limited (%s)",
                    type(exc).__name__,
                )
                raise GenerationUnavailableError("generation_rate_limited") from exc

            if _is_model_unavailable(exc) and self.model != FALLBACK_COHERE_MODEL:
                logger.warning(
                    "Primary Cohere model unavailable (%s); trying fallback",
                    type(exc).__name__,
                )
                try:
                    text = self._chat(prompt, FALLBACK_COHERE_MODEL)
                    self.model = FALLBACK_COHERE_MODEL
                    self.last_error = ""
                    return text
                except Exception as fallback_exc:
                    if _is_rate_limited(fallback_exc):
                        self.last_error = "rate_limited"
                        logger.warning(
                            "Cohere fallback generation rate limited (%s)",
                            type(fallback_exc).__name__,
                        )
                        raise GenerationUnavailableError(
                            "generation_rate_limited"
                        ) from fallback_exc
                    self.last_error = "fallback_unavailable"
                    logger.error(
                        "Cohere fallback generation failed (%s)",
                        type(fallback_exc).__name__,
                    )
                    raise GenerationUnavailableError(
                        "generation_unavailable"
                    ) from fallback_exc

            self.last_error = "generation_unavailable"
            logger.error(
                "Cohere generation failed (%s)",
                type(exc).__name__,
            )
            raise GenerationUnavailableError("generation_unavailable") from exc


class DoctorChatPipeline:
    def __init__(self, uri, user, password, api_key, database=""):
        self.neo4j = Neo4jConnector(uri, user, password, database=database)
        self.rag = BiomedicalRAG(api_key)

    def chat(self, message, conversation):
        context = self.neo4j.search_entities(message)
        history = list(conversation or []) + [{"patient": message}]
        reply = self.rag.answer(history, context)
        return reply, context


_pipeline = None
_pipeline_lock = threading.Lock()


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                neo4j = neo4j_settings()
                _pipeline = DoctorChatPipeline(
                    neo4j["uri"],
                    neo4j["user"],
                    neo4j["password"],
                    cohere_api_key(),
                    database=neo4j["database"],
                )
    return _pipeline


app = Flask(__name__)
CORS(app)

# Reject oversized uploads at the WSGI layer, matching the multimodal API's
# limit so /analyze and /agent behave identically here (oversized -> 413).
app.config["MAX_CONTENT_LENGTH"] = multimodal_api.CONFIG.max_upload_bytes


@app.route("/", methods=["GET"])
def index():
    """Serve the canonical RAGnosis UI.

    This is the original monochrome RAGnosis single-page app (``index.html``):
    the Home / About / Contact pages, the multiround Neo4j+Cohere **Diagnosis**
    chat (POST /chat), and the composed multimodal **Agent** page (POST /agent).
    Both experiences live in one restrained, black-and-white interface.
    """
    return send_from_directory(ROOT_DIR, "index.html")


# Mount the composed multimodal agent + image analysis endpoints by reusing the
# exact, already-tested view functions from multimodal_api (no duplication, no
# forking of the agent/vision/retrieval/health/safety/generation pipeline). They
# read multimodal_api.service / multimodal_api.agent, so their behavior — and the
# test injection of those globals — is preserved verbatim.
app.add_url_rule("/agent", view_func=multimodal_api.agent_analyze, methods=["POST"])
app.add_url_rule("/analyze", view_func=multimodal_api.analyze, methods=["POST"])
app.register_error_handler(413, multimodal_api.too_large)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    conversation = data.get("conversation") or []
    if not message:
        return jsonify({"response": "Please enter a message.", "error": "empty_message"}), 400
    if not isinstance(conversation, list):
        conversation = []

    pipeline = get_pipeline()
    try:
        doctor_reply, context = pipeline.chat(message, conversation)
        return jsonify(
            {
                "response": doctor_reply,
                "disclaimer": MEDICAL_DISCLAIMER,
                "context_found": bool(context and context != "No relevant entities found."),
            }
        )
    except GenerationUnavailableError as exc:
        logger.warning("Chat generation unavailable: %s", exc.code)
        return jsonify(
            {
                "response": SAFE_GENERATION_MESSAGE,
                "error": exc.code,
                "disclaimer": MEDICAL_DISCLAIMER,
            }
        ), 503
    except Exception as exc:
        logger.error("Chat failed (%s)", type(exc).__name__)
        return jsonify(
            {
                "response": SAFE_RETRIEVAL_MESSAGE,
                "error": "retrieval_unavailable",
                "disclaimer": MEDICAL_DISCLAIMER,
            }
        ), 503


@app.route("/health", methods=["GET"])
def health():
    pipeline = get_pipeline()
    neo4j_configured = pipeline.neo4j.configured()
    neo4j_connected = pipeline.neo4j.ping() if neo4j_configured else False
    service = multimodal_api.service
    agent = multimodal_api.agent
    return jsonify(
        {
            "status": "ok",
            "disclaimer": MEDICAL_DISCLAIMER,
            # --- legacy production dependencies (Neo4j + Cohere chat) ---------
            "neo4j": {
                "configured": neo4j_configured,
                "connected": neo4j_connected,
                "database": neo4j_settings()["database"] or None,
                "error": pipeline.neo4j.last_error or None,
            },
            "cohere": {
                "configured": pipeline.rag.configured(),
                "model": pipeline.rag.model,
            },
            # --- new multimodal / agent configuration ------------------------
            # These mirror the fields the agent UI (demo.html) reads from
            # /health so status shows correctly on the canonical entrypoint.
            "vision_configured": service.vision.configured(),
            "generation_configured": service.generator.configured(),
            "health_intelligence_configured": agent.health.configured(),
        }
    )


if __name__ == "__main__":
    port = int(os.environ["PORT"]) if os.environ.get("PORT") else 8000
    app.run(host="0.0.0.0", port=port)
