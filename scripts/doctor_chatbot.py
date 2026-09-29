# ============================================================
# DOCTOR CHATBOT v2.8 — Neo4j + Cohere (Persistent Chat)
# ============================================================

import os
import sys

NEO4J_URI = os.environ.get("NEO4J_URI", "")
NEO4J_USER = os.environ.get("NEO4J_USER", "")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "")

if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, COHERE_API_KEY]):
    print("ERROR: missing environment variables.")
    print("Set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, and COHERE_API_KEY.")
    sys.exit(1)

from neo4j import GraphDatabase
import cohere, json, logging

logging.getLogger("neo4j").setLevel(logging.ERROR)

class Neo4jConnector:
    def __init__(self, uri, user, password, database=""):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        print("Connected to Neo4j Aura")

    def close(self):
        self.driver.close()
        print("Neo4j connection closed")

    def _session(self):
        if self.database:
            return self.driver.session(database=self.database)
        return self.driver.session()

    def search_entities(self, query_text):
        safe_props = ["name", "text", "description", "disease", "symptom", "title", "canonical_name"]
        query_parts = [f"(n.{p} IS NOT NULL AND toLower(toString(n.{p})) CONTAINS toLower($query))" for p in safe_props]
        where_clause = " OR ".join(query_parts)
        cypher_query = f"""
            MATCH (n)
            WHERE {where_clause}
            RETURN n LIMIT 5
        """
        with self._session() as session:
            result = session.run(cypher_query, {"query": query_text})
            records = result.values()
            context = "\n".join([
                json.dumps(
                    {k: v for k, v in r[0]._properties.items() if k.lower() not in {"embedding", "embeddings", "vector"}},
                    indent=2,
                    default=str,
                )
                for r in records
            ])
            return context if context else "No relevant entities found."

class BiomedicalRAG:
    def __init__(self, api_key, model="command-a-03-2025"):
        self.cohere = cohere.Client(api_key)
        self.model = model
        print(f"Cohere Chat backend initialized with model '{self.model}'")

    def _prompt(self, conversation, context):
        convo_text = "\n".join([
            f"Patient: {c['patient']}\nDoctor: {c['doctor']}" if 'doctor' in c else f"Patient: {c['patient']}"
            for c in conversation
        ])
        return f"""
You are a kind, logical, biomedical doctor chatbot.

SAFETY:
- You are an informational assistant, not a licensed clinician.
- Do not claim to replace professional diagnosis or treatment.

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
    def __init__(self, uri, user, password, api_key, database=""):
        self.neo4j = Neo4jConnector(uri, user, password, database=database)
        self.rag = BiomedicalRAG(api_key)
        print("Doctor Chatbot Ready")

    def chat(self):
        print("\nDoctorBot: Hello! Please tell me how you're feeling today.")
        print("This assistant is informational and does not replace professional medical care.")
        conversation = []
        diagnosis_suggested = False

        while True:
            patient_input = input("\nYou: ").strip()
            if patient_input.lower() in ["quit", "exit", "bye"]:
                print("\nDoctorBot: Take care! Wishing you good health.")
                break

            conversation.append({"patient": patient_input})
            context = self.neo4j.search_entities(patient_input)
            doctor_reply = self.rag.answer(conversation, context)
            print(f"\nDoctorBot: {doctor_reply}\n")

            conversation[-1]["doctor"] = doctor_reply

            if any(keyword in doctor_reply.lower() for keyword in [
                "diagnosis", "you may have", "it appears", "likely cause", "seems like"
            ]):
                if not diagnosis_suggested:
                    print("\nDoctorBot: I've suggested a likely diagnosis. "
                          "Continue chatting or type 'quit' to end.")
                diagnosis_suggested = True

        print("\nConversation Summary")
        print("=" * 60)
        for turn in conversation:
            print(f"You: {turn['patient']}")
            if "doctor" in turn:
                print(f"DoctorBot: {turn['doctor']}\n")
        print("=" * 60)

if __name__ == "__main__":
    pipeline = DoctorChatPipeline(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, COHERE_API_KEY, database=NEO4J_DATABASE)
    pipeline.chat()
