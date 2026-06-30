import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from dotenv import load_dotenv
load_dotenv()
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

EMBED_MODEL = "text-embedding-3-small"
COLLECTION = "session3_secure_kb"
DEFAULT_DSN = "postgresql+psycopg://rag:rag@localhost:5433/ragdb"


def die(msg: str):
    print(f"\n[setup] {msg}", file=sys.stderr)
    sys.exit(1)


def pg_connection() -> str:
    dsn = os.getenv("DATABASE_URL", DEFAULT_DSN)
    if dsn.startswith("postgresql://"):
        dsn = dsn.replace("postgresql://", "postgresql+psycopg://", 1)
    return dsn


def get_embedder() -> OpenAIEmbeddings:
    if not os.getenv("OPENAI_API_KEY"):
        die("OPENAI_API_KEY is not set.")
    return OpenAIEmbeddings(model=EMBED_MODEL)


def load_chunks(path: str) -> list[dict]:
    if not os.path.exists(path):
        die(f"{path} not found. Run: python build_corpus_learner.py")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def to_documents(chunks: list[dict]) -> list[Document]:
    return [
        Document(
            page_content=c["content"],
            metadata={
                "source": c.get("source"),
                "section": c.get("section"),
                "chunk_id": c.get("chunk_id"),
                "document_type": c.get("document_type"),
                "department": c.get("department", "all"),
                "access_level": c.get("access_level", "public"),
                "tenant_id": c.get("tenant_id", "acme"),
                #"object_location": c.get("object_location", "s3://rag-bucket/"),
            },
        )
        for c in chunks
    ]


@dataclass
class UserContext:
    name: str
    role: str
    department: str
    access_levels: list = field(default_factory=lambda: ["public"])
    tenant_id: str = "acme"


USERS = {
    "admin": UserContext("Asha (admin)", "admin", "all", ["public", "internal", "restricted"]),
    "dept": UserContext("Dev (engineering)", "employee", "engineering", ["public", "internal"]),
    "public": UserContext("Pat (public)", "public", "none", ["public"]),
}


def access_filter(user: UserContext) -> dict:
    clauses = [
        {"tenant_id": {"$eq": user.tenant_id}},
        {"access_level": {"$in": user.access_levels}},
    ]
    if user.role != "admin":
        clauses.append({"department": {"$in": ["all", user.department]}})
    return {"$and": clauses}


def rule(s):
    print("\n" + "=" * 72 + "\n" + s + "\n" + "=" * 72)


def main(query):
    chunks = load_chunks("RAG-Access-Control\\data\\chunks.json")
    docs = to_documents(chunks)
    emb = get_embedder()

    rule(f"BUILD  (PGVector '{COLLECTION}')")
    try:
        store = PGVector(
            embeddings=emb,
            collection_name=COLLECTION,
            connection=pg_connection(),
            use_jsonb=True,
            pre_delete_collection=True,
        )
    except Exception as e:
        die(f"cannot connect to Postgres ({e}). Start: docker compose up -d")
    store.add_documents(docs)
    n_restricted = sum(1 for c in chunks if c["access_level"] == "restricted")
    print(f"   stored {len(docs)} chunks ({n_restricted} restricted)")

    rule(f"SAME QUERY, THREE USERS\n   query: {query!r}")
    leaked = False
    for key, user in USERS.items():
        flt = access_filter(user)
        hits = store.similarity_search(query, k=3, filter=flt)
        print(f"\n--- {user.name}  (role={user.role}, dept={user.department}) ---")
        if not hits:
            print("    (no permitted results)")
        for rank, d in enumerate(hits, 1):
            lvl = d.metadata["access_level"]
            print(
                f"    {rank}. [{lvl.upper():10s}] {d.metadata['section']:26s} "
                f"({d.metadata['source']})"
            )
            if lvl == "restricted" and key != "admin":
                leaked = True

    rule("VERDICT")
    if leaked:
        print("   [!!] Restricted content reached a non-admin user.")
    else:
        print("   [ok] Restricted chunks returned only to admin.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--query",
        default="What is the compensation structure for senior leadership?",
    )
    main(ap.parse_args().query)
