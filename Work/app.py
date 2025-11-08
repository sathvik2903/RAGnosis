from flask import Flask, request, jsonify
from flask_cors import CORS
from neo4j import GraphDatabase
import cohere, json, logging

# --- Hide Neo4j debug output ---
logging.getLogger("neo4j").setLevel(logging.ERROR)

# --- Credentials (use your own secure method in production) ---
NEO4J_URI = "neo4j+s://62418b31.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "RkWEB1D_av8SZWSgbVqM7kCLOeo5DZTSZJHWW5lECZo"
COHERE_KEY = "aO28RpD2lrgCB6DLYIXYWuNXetIishEvcKRsXzB5"

# --- MRD-RAG Classes ---

class Neo4jConnector:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def search_entities(self, query_text):
        safe_props = ["name", "text", "description", "disease", "symptom", "title"]
        query_parts = [f"(n.{p} IS NOT NULL AND toLower(toString(n.{p})) CONTAINS toLower($query))" for p in safe_props]
        where_clause = " OR ".join(query_parts)
        cypher_query = f"""
            MATCH (n)
            WHERE {where_clause}
            RETURN n LIMIT 5
        """
        with self.driver.session() as session:
            result = session.run(cypher_query, {"query": query_text})
            records = result.values()
            context = "\n".join([
                json.dumps(r[0]._properties, indent=2)
                for r in records
            ])
            return context if context else "No relevant entities found."

class BiomedicalRAG:
    def __init__(self, api_key, model="command-a-03-2025"):
        self.cohere = cohere.Client(api_key)
        self.model = model
        try:
            _ = self.cohere.chat(model=self.model, message="Hello test")
        except Exception:
            self.model = "command-r7b-12-2024"

    def _prompt(self, conversation, context):
        convo_text = "\n".join([
            f"Patient: {c['patient']}\nDoctor: {c.get('doctor', '')}" if 'doctor' in c else f"Patient: {c['patient']}"
            for c in conversation
        ])
        return f"""
You are a kind, logical, biomedical doctor chatbot.

PATIENT CONVERSATION HISTORY:
{convo_text}

KNOWLEDGE GRAPH CONTEXT:
{context}

TASK:
- Ask relevant medical follow-up questions.
- If enough info, provide the most likely diagnosis and advice.
- Be concise, empathetic, and medically accurate.
- End the chat when confident in your diagnosis.

YOUR RESPONSE:
"""

    def answer(self, conversation, context):
        try:
            response = self.cohere.chat(
                model=self.model,
                message=self._prompt(conversation, context),
                temperature=0.4
            )
            return response.text.strip()
        except Exception as e:
            return f"Error querying Cohere Chat API: {e}"

class DoctorChatPipeline:
    def __init__(self, uri, user, password, api_key):
        self.neo4j = Neo4jConnector(uri, user, password)
        self.rag = BiomedicalRAG(api_key)

# --- Flask Web Server ---

app = Flask(__name__)
CORS(app)

pipeline = DoctorChatPipeline(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, COHERE_KEY)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get("message", "")
    conversation = data.get("conversation", [])
    context = pipeline.neo4j.search_entities(message)
    doctor_reply = pipeline.rag.answer(conversation + [{'patient': message}], context)
    return jsonify({
        "response": doctor_reply
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    # The port will be set by Render or Heroku, fallback to 10000 locally
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
