from pathlib import Path
import os

from neo4j import GraphDatabase
import pandas as pd
from tqdm import tqdm

NEO4J_URI = os.environ.get("NEO4J_URI", "")
NEO4J_USER = os.environ.get("NEO4J_USER", "")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "")

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = Path(os.environ.get(
    "ENRICHMENT_DIR",
    REPO_ROOT / "Database" / "EnrichmentReport",
))
SIM_LINKS_FILE = BASE_DIR / "sim_links.csv"

if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
    raise RuntimeError("Set NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD before running ingestion.")

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

def _session():
    if NEO4J_DATABASE:
        return driver.session(database=NEO4J_DATABASE)
    return driver.session()

print("Connected to Neo4j Aura")

with _session() as s:
    s.run("""
    CREATE CONSTRAINT IF NOT EXISTS
    FOR (c:EntityCanonical)
    REQUIRE c.canonical_id IS UNIQUE
    """)
print("Constraints ensured")

sim_df = pd.read_csv(SIM_LINKS_FILE)

def pick(colnames):
    for c in sim_df.columns:
        for k in colnames:
            if k in c.lower():
                return c
    return None

src_col = pick(["src", "source"])
tgt_col = pick(["tgt", "target"])
score_col = pick(["score"])

if not src_col or not tgt_col or not score_col:
    raise RuntimeError(f"Unsupported sim_links schema: {sim_df.columns.tolist()}")

print("Using sim_links columns:", src_col, tgt_col, score_col)

with _session() as s:
    for _, r in tqdm(sim_df.iterrows(), total=len(sim_df), desc="SIMILAR_TO links"):
        s.run(
            """
            MATCH (a:EntityCanonical {canonical_id:$a})
            MATCH (b:EntityCanonical {canonical_id:$b})
            MERGE (a)-[rel:SIMILAR_TO]->(b)
            SET rel.score = $score
            """,
            a=str(r[src_col]),
            b=str(r[tgt_col]),
            score=float(r[score_col])
        )

print("Similarity ingestion completed")

with _session() as s:
    stats = {
        "canonical_nodes": s.run(
            "MATCH (c:EntityCanonical) RETURN count(c) AS c"
        ).single()["c"],
        "similarity_links": s.run(
            "MATCH (:EntityCanonical)-[r:SIMILAR_TO]->(:EntityCanonical) RETURN count(r) AS c"
        ).single()["c"],
        "canonical_without_similarity": s.run(
            """
            MATCH (c:EntityCanonical)
            WHERE NOT (c)-[:SIMILAR_TO]->()
              AND NOT ()-[:SIMILAR_TO]->(c)
            RETURN count(c) AS c
            """
        ).single()["c"]
    }

print("\n===== INTEGRITY REPORT =====")
for k, v in stats.items():
    print(f"{k}: {v}")

driver.close()
print("\nContinuation pipeline completed cleanly")
