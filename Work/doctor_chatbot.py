# ============================================================
# 🩺 DOCTOR CHATBOT v2.8 — Neo4j + Cohere (2025, Persistent Chat)
# ============================================================

# Import credentials from config.py
try:
    from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, COHERE_KEY
except ImportError:
    print("❌ ERROR: config.py not found!")
    print("Create config.py from config.example.py and add your credentials.")
    exit(1)

# Import Libraries
from neo4j import GraphDatabase
import cohere, json, logging

# Hide Neo4j debug output
logging.getLogger("neo4j").setLevel(logging.ERROR)

# Neo4j Connector (safe query filtering)
class Neo4jConnector:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        print(f"✓ Connected to Neo4j at {uri}")

    def close(self):
        self.driver.close()
        print("✓ Neo4j connection closed")

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

# Cohere Biomedical Chat Backend
class BiomedicalRAG:
    def __init__(self, api_key, model="command-r-08-2024"):
        self.cohere = cohere.Client(api_key)
        self.model = model
        print(f"✓ Cohere Chat backend initialized with model '{self.model}'")

    def _prompt(self, conversation, context):
        convo_text = "\n".join([
            f"Patient: {c['patient']}\nDoctor: {c['doctor']}" if 'doctor' in c else f"Patient: {c['patient']}"
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

# Doctor Chatbot Pipeline
class DoctorChatPipeline:
    def __init__(self, uri, user, password, api_key):
        self.neo4j = Neo4jConnector(uri, user, password)
        self.rag = BiomedicalRAG(api_key)
        print("✓ Doctor Chatbot Ready")

    def chat(self):
        print("\n👨‍⚕️ DoctorBot: Hello! Please tell me how you're feeling today.")
        conversation = []
        diagnosis_suggested = False

        while True:
            patient_input = input("\n🧍‍♂️ You: ").strip()
            if patient_input.lower() in ["quit", "exit", "bye"]:
                print("\n👨‍⚕️ DoctorBot: Take care! Wishing you good health.")
                break

            conversation.append({"patient": patient_input})
            context = self.neo4j.search_entities(patient_input)
            doctor_reply = self.rag.answer(conversation, context)
            print(f"\n👨‍⚕️ DoctorBot: {doctor_reply}\n")

            conversation[-1]["doctor"] = doctor_reply

            if any(keyword in doctor_reply.lower() for keyword in [
                "diagnosis", "you may have", "it appears", "likely cause", "seems like"
            ]):
                if not diagnosis_suggested:
                    print("\n✅ DoctorBot: I've suggested a likely diagnosis. "
                          "Continue chatting or type 'quit' to end.")
                diagnosis_suggested = True

        # Summary
        print("\n🩺 Conversation Summary")
        print("=" * 60)
        for turn in conversation:
            print(f"🧍‍♂️ You: {turn['patient']}")
            if "doctor" in turn:
                print(f"👨‍⚕️ DoctorBot: {turn['doctor']}\n")
        print("=" * 60)

# Start Interactive Chat
if __name__ == "__main__":
    pipeline = DoctorChatPipeline(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, COHERE_KEY)
    pipeline.chat()
