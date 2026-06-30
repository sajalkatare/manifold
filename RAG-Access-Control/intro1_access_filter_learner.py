"""Access filter demo: same query, different metadata filters.

Run:
    docker compose up -d
    export OPENAI_API_KEY=sk-...
    python intro1_access_filter_learner.py
"""

import os
import sys
from dotenv import load_dotenv
load_dotenv()
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

if not os.getenv("OPENAI_API_KEY"):
    print("[setup] export OPENAI_API_KEY=sk-... first.", file=sys.stderr)
    sys.exit(1)

dsn = os.getenv("DATABASE_URL", "postgresql+psycopg://rag:rag@localhost:5433/ragdb")
if dsn.startswith("postgresql://"):
    dsn = dsn.replace("postgresql://", "postgresql+psycopg://", 1)

print("=" * 64)
print("Access filter — same query, different users")
print("=" * 64)

docs = [
    Document(
        page_content="Wellness program reimburses gym memberships up to 50 USD/month.",
        metadata={"section": "Wellness", "access_level": "public"},
    ),
    Document(
        page_content="VPN error ERR-5021 means authentication failure; reset VPN-7.",
        metadata={"section": "VPN Errors", "access_level": "internal"},
    ),
    Document(
        page_content="CEO base salary is 480,000 USD with a bonus up to 100 percent.",
        metadata={"section": "CEO Package", "access_level": "restricted"},
    ),
]

emb = OpenAIEmbeddings(model="text-embedding-3-small")
try:
    store = PGVector(
        embeddings=emb,
        collection_name="intro_access_demo",
        connection=dsn,
        use_jsonb=True,
        pre_delete_collection=True,
    )
except Exception as e:
    print(f"[setup] cannot connect to Postgres ({e}). Start: docker compose up -d", file=sys.stderr)
    sys.exit(1)

store.add_documents(docs)
print(f"\nStored {len(docs)} docs.")

query = "How much does leadership get paid?"
print(f"\nQuery: {query!r}")

public_filter = {"access_level": {"$in": ["public"]}}
admin_filter = {"access_level": {"$in": ["public", "internal", "restricted"]}}

for label, flt in [("PUBLIC user", public_filter), ("ADMIN user", admin_filter)]:
    print(f"\n--- {label}  filter={flt} ---")
    hits = store.similarity_search(query, k=3, filter=flt)
    for rank, d in enumerate(hits, 1):
        print(f"  {rank}. [{d.metadata['access_level']:10s}] {d.metadata['section']}")
