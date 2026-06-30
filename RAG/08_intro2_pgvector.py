"""
intro2_pgvector.py — WARM-UP before 02_build_kb.py
==================================================
The smallest possible LangChain + PGVector example. Three toy documents,
one search — so the vector store stops being abstract before the full build.

  - a chunk becomes a LangChain Document (text + metadata)
  - PGVector stores text + embedding + metadata in one row
  - similarity_search_with_score returns the closest doc + a distance

Run:
    docker compose up -d
    export OPENAI_API_KEY=sk-...
    python intro2_pgvector.py

Then run the real thing: python 02_build_kb.py
"""
import os
import sys
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector
from dotenv import load_dotenv
load_dotenv()

dsn = os.getenv("DATABASE_URL", "postgresql+psycopg://rag:rag@localhost:5433/ragdb")

docs = [
    Document(page_content="Employees get 26 weeks of parental leave per child.",
             metadata={"section": "Parental Leave"}),
    Document(page_content="You may work remotely up to three days per week.",
             metadata={"section": "Remote Work"}),
    Document(page_content="Multi-factor authentication is mandatory for all systems.",
             metadata={"section": "MFA"}),
]

emb = OpenAIEmbeddings(model="text-embedding-3-small")
try:
    store = PGVector(embeddings=emb, collection_name="intro_demo",
                     connection=dsn, use_jsonb=True, pre_delete_collection=True)
except Exception as e:
    print(f"[setup] cannot connect to Postgres ({e}). Start: docker compose up -d",
          file=sys.stderr)
    sys.exit(1)

# No custom ids -> LangChain assigns UUIDs (ids are global across collections).
store.add_documents(docs)
print(f"\nStored {len(docs)} documents in pgvector (collection 'intro_demo').")

query = "How many weeks off for a new baby?"
print(f"\nQuery: {query!r}\n")
for rank, (doc, dist) in enumerate(store.similarity_search_with_score(query, k=3), 1):
    print(f"  {rank}. dist={dist:.3f}  [{doc.metadata['section']}]  {doc.page_content}")

print("\nClosest = smallest distance, and it's the parental-leave doc — even though")

